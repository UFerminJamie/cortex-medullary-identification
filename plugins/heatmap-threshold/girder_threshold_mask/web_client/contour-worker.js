importScripts('polygon-clipping.umd.min.js');

const clipping = self.polygonClipping;
const BLOCK_SIZE = 256;
const MAX_TREE_CACHE_NODES = 512;
const MAX_TREE_CACHE_COORDS = 2000000;
const MAX_RESULT_CACHE = 12;
const MAX_RESULT_CACHE_COORDS = 1000000;

let polygons = [];
let scores = new Float64Array(0);
let treeBase = 1;
let blockCount = 0;
let treeCache = new Map();
let treeCacheCoords = 0;
// Cache unsimplified geometries only. Rendering settings never poison this cache.
let resultCache = new Map();
let resultCacheCoords = 0;
let warmTimer = null;
let warmBlocks = [];
let sourceOptions = {snap: 0, gapTolerance: 0};
let pendingThreshold = null;
let thresholdTimer = null;

function snapCoordinate(value, snap) {
    return snap > 0 ? Math.round(value / snap) * snap : value;
}

function signedArea(ring) {
    let area = 0;
    for (let i = 0; i < ring.length - 1; i += 1) {
        area += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1];
    }
    return area / 2;
}

// Positive distance offsets away from the ring's geometric interior.
function offsetRing(ring, distance) {
    if (!distance || ring.length < 4) {
        return ring.map((point) => point.slice());
    }
    const open = ring.slice(0, -1);
    const orientation = signedArea(ring) >= 0 ? 1 : -1;
    const output = [];
    for (let i = 0; i < open.length; i += 1) {
        const previous = open[(i + open.length - 1) % open.length];
        const current = open[i];
        const next = open[(i + 1) % open.length];
        const e1x = current[0] - previous[0];
        const e1y = current[1] - previous[1];
        const e2x = next[0] - current[0];
        const e2y = next[1] - current[1];
        const l1 = Math.hypot(e1x, e1y) || 1;
        const l2 = Math.hypot(e2x, e2y) || 1;
        // Right-hand normals, corrected for ring winding.
        const n1x = orientation * e1y / l1;
        const n1y = -orientation * e1x / l1;
        const n2x = orientation * e2y / l2;
        const n2y = -orientation * e2x / l2;
        const p1x = current[0] + n1x * distance;
        const p1y = current[1] + n1y * distance;
        const p2x = current[0] + n2x * distance;
        const p2y = current[1] + n2y * distance;
        const cross = e1x * e2y - e1y * e2x;
        if (Math.abs(cross) < 1e-12) {
            const nx = n1x + n2x;
            const ny = n1y + n2y;
            const length = Math.hypot(nx, ny) || 1;
            const scale = Math.min(Math.abs(distance) * 4, Math.abs(distance) / Math.max(0.25, (nx * n1x + ny * n1y) / length));
            output.push([current[0] + nx / length * scale, current[1] + ny / length * scale]);
        } else {
            const t = ((p2x - p1x) * e2y - (p2y - p1y) * e2x) / cross;
            const x = p1x + t * e1x;
            const y = p1y + t * e1y;
            const miter = Math.hypot(x - current[0], y - current[1]);
            const maxMiter = Math.max(Math.abs(distance) * 8, Math.abs(distance));
            if (miter > maxMiter) {
                const nx = n1x + n2x;
                const ny = n1y + n2y;
                const length = Math.hypot(nx, ny) || 1;
                output.push([current[0] + nx / length * Math.abs(distance), current[1] + ny / length * Math.abs(distance)]);
            } else {
                output.push([x, y]);
            }
        }
    }
    if (output.length) {
        output.push(output[0].slice());
    }
    return output;
}

function offsetGeometry(geometry, distance) {
    if (!geometry || !distance) {
        return geometry;
    }
    return geometry.map((polygon) => polygon.map((ring, index) => {
        // Dilating a filled polygon grows exteriors and contracts holes.
        return offsetRing(ring, index === 0 ? distance : -distance);
    }));
}

function snapRing(ring, snap) {
    const result = ring.map(([x, y]) => [snapCoordinate(x, snap), snapCoordinate(y, snap)]);
    if (result.length > 1) {
        result[result.length - 1] = result[0].slice();
    }
    return result;
}

function toPolygon(element, snap = 0, gapTolerance = 0) {
    let polygon;
    if (element.type === 'rectangle') {
        const [x, y] = element.center;
        const halfWidth = element.width / 2;
        const halfHeight = element.height / 2;
        const angle = (element.rotation || 0) * Math.PI / 180;
        const corners = [
            [-halfWidth, -halfHeight], [halfWidth, -halfHeight],
            [halfWidth, halfHeight], [-halfWidth, halfHeight]
        ].map(([dx, dy]) => [
            x + dx * Math.cos(angle) - dy * Math.sin(angle),
            y + dx * Math.sin(angle) + dy * Math.cos(angle)
        ]);
        corners.push(corners[0].slice());
        polygon = [corners];
    } else if (element.type === 'polyline' && element.closed && element.points && element.points.length >= 3) {
        const outer = element.points.map(([x, y]) => [x, y]);
        if (outer[0][0] !== outer[outer.length - 1][0] || outer[0][1] !== outer[outer.length - 1][1]) {
            outer.push(outer[0].slice());
        }
        polygon = [outer, ...(element.holes || []).map((hole) => {
            const ring = hole.map(([x, y]) => [x, y]);
            if (ring.length && (ring[0][0] !== ring[ring.length - 1][0] || ring[0][1] !== ring[ring.length - 1][1])) {
                ring.push(ring[0].slice());
            }
            return ring;
        })];
    } else {
        return null;
    }
    if (snap > 0) {
        polygon = polygon.map((ring) => snapRing(ring, snap));
    }
    if (gapTolerance > 0) {
        polygon = offsetGeometry([polygon], gapTolerance / 2)[0];
    }
    return [polygon];
}

function geometryCoordinates(geometry) {
    let count = 0;
    if (geometry) {
        geometry.forEach((polygon) => polygon.forEach((ring) => { count += ring.length; }));
    }
    return count;
}

function remember(map, key, geometry, coordinateCount, limits) {
    if (!geometry || coordinateCount > limits.coordinates) {
        return;
    }
    const previous = map.get(key);
    if (previous) {
        if (previous.geometry === geometry) {
            map.delete(key);
            map.set(key, previous);
            return;
        }
        limits.current -= previous.coordinateCount;
    }
    map.set(key, {geometry, coordinateCount});
    limits.current += coordinateCount;
    while (map.size > limits.entries || limits.current > limits.coordinates) {
        const oldestKey = map.keys().next().value;
        const oldest = map.get(oldestKey);
        map.delete(oldestKey);
        limits.current -= oldest.coordinateCount;
    }
}

function cached(map, key) {
    const entry = map.get(key);
    if (!entry) {
        return null;
    }
    map.delete(key);
    map.set(key, entry);
    return entry.geometry;
}

function unionGeometries(geometries) {
    const valid = geometries.filter(Boolean);
    if (!valid.length) {
        return null;
    }
    if (valid.length === 1) {
        return valid[0];
    }
    const batches = [];
    for (let i = 0; i < valid.length; i += 64) {
        batches.push(clipping.union(...valid.slice(i, i + 64)));
    }
    return batches.length === 1 ? batches[0] : clipping.union(...batches);
}

function unionPolygonRange(start, end) {
    if (end <= start) {
        return null;
    }
    const batch = [];
    for (let i = start; i < end; i += 1) {
        batch.push(polygons[i]);
    }
    return unionGeometries(batch);
}

function getTreeNode(index) {
    const existing = cached(treeCache, index);
    if (existing) {
        return existing;
    }
    let geometry;
    if (index >= treeBase) {
        const blockIndex = index - treeBase;
        geometry = blockIndex < blockCount
            ? unionPolygonRange(blockIndex * BLOCK_SIZE, Math.min(polygons.length, (blockIndex + 1) * BLOCK_SIZE))
            : null;
    } else {
        geometry = unionGeometries([getTreeNode(index * 2), getTreeNode(index * 2 + 1)]);
    }
    if (geometry) {
        const coordinateCount = geometryCoordinates(geometry);
        remember(treeCache, index, geometry, coordinateCount, {
            entries: MAX_TREE_CACHE_NODES,
            coordinates: MAX_TREE_CACHE_COORDS,
            get current() { return treeCacheCoords; },
            set current(value) { treeCacheCoords = value; }
        });
    }
    return geometry;
}

function queryFullBlocks(startBlock, endBlock) {
    const pieces = [];
    let left = startBlock + treeBase;
    let right = endBlock + treeBase;
    while (left < right) {
        if (left & 1) {
            pieces.push(getTreeNode(left));
            left += 1;
        }
        if (right & 1) {
            right -= 1;
            pieces.push(getTreeNode(right));
        }
        left >>= 1;
        right >>= 1;
    }
    return pieces;
}

function unionRange(start, end) {
    if (end <= start) {
        return null;
    }
    const firstBoundary = Math.ceil(start / BLOCK_SIZE) * BLOCK_SIZE;
    if (firstBoundary >= end) {
        return unionPolygonRange(start, end);
    }
    const fullStart = firstBoundary / BLOCK_SIZE;
    const fullEnd = end === polygons.length ? blockCount : Math.floor(end / BLOCK_SIZE);
    const pieces = [];
    const leading = unionPolygonRange(start, firstBoundary);
    if (leading) {
        pieces.push(leading);
    }
    pieces.push(...queryFullBlocks(fullStart, fullEnd));
    const trailingStart = fullEnd * BLOCK_SIZE;
    const trailing = trailingStart < end ? unionPolygonRange(trailingStart, end) : null;
    if (trailing) {
        pieces.push(trailing);
    }
    return unionGeometries(pieces);
}

function splitForThreshold(threshold) {
    let low = 0;
    let high = scores.length;
    while (low < high) {
        const middle = (low + high) >>> 1;
        if (scores[middle] >= threshold) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low;
}

function geometriesAt(split) {
    let geometries = cached(resultCache, split);
    if (!geometries) {
        geometries = {
            medulla: unionRange(0, split),
            cortex: unionRange(split, polygons.length)
        };
        const coordinateCount = geometryCoordinates(geometries.cortex) + geometryCoordinates(geometries.medulla);
        remember(resultCache, split, geometries, coordinateCount, {
            entries: MAX_RESULT_CACHE,
            coordinates: MAX_RESULT_CACHE_COORDS,
            get current() { return resultCacheCoords; },
            set current(value) { resultCacheCoords = value; }
        });
    }
    return geometries;
}

function perpendicularDistanceSquared(point, start, end) {
    const dx = end[0] - start[0];
    const dy = end[1] - start[1];
    if (!dx && !dy) {
        const px = point[0] - start[0];
        const py = point[1] - start[1];
        return px * px + py * py;
    }
    const cross = (point[0] - start[0]) * dy - (point[1] - start[1]) * dx;
    return cross * cross / (dx * dx + dy * dy);
}

function simplifyOpen(points, tolerance) {
    if (points.length <= 2 || tolerance <= 0) {
        return points.map((point) => point.slice());
    }
    const toleranceSquared = tolerance * tolerance;
    const keep = new Uint8Array(points.length);
    keep[0] = 1;
    keep[points.length - 1] = 1;
    const stack = [[0, points.length - 1]];
    while (stack.length) {
        const [start, end] = stack.pop();
        let maxDistance = toleranceSquared;
        let index = -1;
        for (let i = start + 1; i < end; i += 1) {
            const distance = perpendicularDistanceSquared(points[i], points[start], points[end]);
            if (distance > maxDistance) {
                maxDistance = distance;
                index = i;
            }
        }
        if (index !== -1) {
            keep[index] = 1;
            stack.push([start, index], [index, end]);
        }
    }
    const simplified = [];
    for (let i = 0; i < points.length; i += 1) {
        if (keep[i]) {
            simplified.push(points[i].slice());
        }
    }
    return simplified;
}

function simplifyClosedRing(ring, tolerance) {
    if (tolerance <= 0 || ring.length < 5) {
        return ring.map((point) => point.slice());
    }
    const open = ring.slice(0, -1);
    if (open.length < 4) {
        return ring.map((point) => point.slice());
    }
    // Select the widest pair among axis extrema; this is stable under ring rotation.
    const extrema = [0, 0, 0, 0]; // min x, max x, min y, max y
    for (let i = 1; i < open.length; i += 1) {
        if (open[i][0] < open[extrema[0]][0]) { extrema[0] = i; }
        if (open[i][0] > open[extrema[1]][0]) { extrema[1] = i; }
        if (open[i][1] < open[extrema[2]][1]) { extrema[2] = i; }
        if (open[i][1] > open[extrema[3]][1]) { extrema[3] = i; }
    }
    let anchorA = extrema[0];
    let anchorB = extrema[1];
    let widest = -1;
    for (let i = 0; i < extrema.length; i += 1) {
        for (let j = i + 1; j < extrema.length; j += 1) {
            const dx = open[extrema[i]][0] - open[extrema[j]][0];
            const dy = open[extrema[i]][1] - open[extrema[j]][1];
            const distance = dx * dx + dy * dy;
            if (distance > widest) {
                widest = distance;
                anchorA = extrema[i];
                anchorB = extrema[j];
            }
        }
    }
    const firstArc = [];
    for (let i = anchorA; ; i = (i + 1) % open.length) {
        firstArc.push(open[i]);
        if (i === anchorB) {
            break;
        }
    }
    const secondArc = [];
    for (let i = anchorB; ; i = (i + 1) % open.length) {
        secondArc.push(open[i]);
        if (i === anchorA) {
            break;
        }
    }
    let simplified = simplifyOpen(firstArc, tolerance).slice(0, -1)
        .concat(simplifyOpen(secondArc, tolerance).slice(0, -1));
    const originalArea = signedArea(ring);
    const candidate = simplified.concat([simplified[0] && simplified[0].slice()]);
    const simplifiedArea = candidate.length >= 4 ? signedArea(candidate) : 0;
    // Keep the original ring if simplification collapses it or flips its winding.
    if (candidate.length < 4 || Math.abs(simplifiedArea) < Math.max(1e-12, Math.abs(originalArea) * 1e-6) || originalArea * simplifiedArea <= 0) {
        return ring.map((point) => point.slice());
    }
    // polygon-clipping normalization detects candidate rings that changed topology or area.
    try {
        const normalized = clipping.union([[candidate]]);
        if (normalized.length !== 1 || normalized[0].length !== 1 ||
            Math.abs(Math.abs(signedArea(candidate)) - componentArea(normalized[0])) >
                Math.max(1e-8, Math.abs(signedArea(candidate)) * 1e-6)) {
            return ring.map((point) => point.slice());
        }
    } catch (error) {
        return ring.map((point) => point.slice());
    }
    return candidate;
}

function componentArea(polygon) {
    return Math.max(0, Math.abs(signedArea(polygon[0])) - polygon.slice(1).reduce((sum, hole) => sum + Math.abs(signedArea(hole)), 0));
}

function toElements(geometry, options, kind) {
    if (!geometry) {
        return [];
    }
    let outputGeometry = geometry;
    if (options.gapTolerance > 0) {
        try {
            outputGeometry = clipping.union(offsetGeometry(geometry, -options.gapTolerance / 2));
        } catch (error) {
            // Retain the valid grown union if an erosion collapses a narrow feature.
            outputGeometry = geometry;
        }
    }
    const result = [];
    const rgb = kind === 'medulla' ? '255, 0, 0' : '0, 0, 255';
    outputGeometry.forEach((polygon) => {
        if (componentArea(polygon) < options.minArea) {
            return;
        }
        let outer = simplifyClosedRing(polygon[0], options.simplifyTolerance);
        if (outer.length < 4) {
            return;
        }
        const holes = [];
        polygon.slice(1).forEach((hole) => {
            const simplified = simplifyClosedRing(hole, options.simplifyTolerance);
            if (simplified.length >= 4 && Math.abs(signedArea(simplified)) > 1e-12) {
                holes.push(simplified.map(([x, y]) => [x, y, 0]));
            }
        });
        result.push({
            type: 'polyline',
            points: outer.map(([x, y]) => [x, y, 0]),
            holes,
            closed: true,
            fillColor: options.fillColor || `rgba(${rgb}, ${Number.isFinite(options.opacity) ? options.opacity : 0.3})`,
            lineColor: options.lineColor || `rgb(${rgb})`,
            lineWidth: options.lineWidth
        });
    });
    return result;
}

function contoursAt(split, options) {
    const geometry = geometriesAt(split);
    return {
        cortex: toElements(geometry.cortex, options, 'cortex'),
        medulla: toElements(geometry.medulla, options, 'medulla')
    };
}

function scheduleWarmup(split) {
    if (warmTimer !== null) {
        clearTimeout(warmTimer);
    }
    const center = Math.min(blockCount - 1, Math.floor(split / BLOCK_SIZE));
    warmBlocks = [];
    for (let distance = 0; distance <= 4; distance += 1) {
        if (center - distance >= 0) {
            warmBlocks.push(center - distance);
        }
        if (distance && center + distance < blockCount) {
            warmBlocks.push(center + distance);
        }
    }
    const warmNext = () => {
        const block = warmBlocks.shift();
        if (block === undefined) {
            warmTimer = null;
            return;
        }
        getTreeNode(treeBase + block);
        warmTimer = setTimeout(warmNext, 0);
    };
    warmTimer = setTimeout(warmNext, 0);
}

function clearCaches() {
    treeCache = new Map();
    treeCacheCoords = 0;
    resultCache = new Map();
    resultCacheCoords = 0;
}

function loadSources(message) {
    if (warmTimer !== null) {
        clearTimeout(warmTimer);
        warmTimer = null;
    }
    if (thresholdTimer !== null) {
        clearTimeout(thresholdTimer);
        thresholdTimer = null;
    }
    clearCaches();
    const snap = Number.isFinite(message.snap) && message.snap > 0 ? message.snap : 0;
    const gapTolerance = Number.isFinite(message.gapTolerance) && message.gapTolerance > 0 ? message.gapTolerance : 0;
    sourceOptions = {snap, gapTolerance};
    const entries = [];
    (message.elements || []).forEach((element) => {
        const score = (element.user || {}).decision_score;
        const polygon = Number.isFinite(score) ? toPolygon(element, snap, gapTolerance) : null;
        if (polygon) {
            entries.push({score, polygon});
        }
    });
    entries.sort((left, right) => right.score - left.score);
    scores = new Float64Array(entries.length);
    polygons = new Array(entries.length);
    entries.forEach((entry, index) => {
        scores[index] = entry.score;
        polygons[index] = entry.polygon;
    });
    blockCount = Math.ceil(polygons.length / BLOCK_SIZE);
    treeBase = 1;
    while (treeBase < blockCount) {
        treeBase <<= 1;
    }
    const returnedScores = scores.slice();
    self.postMessage({type: 'ready', count: polygons.length, scores: returnedScores.buffer}, [returnedScores.buffer]);
    if (blockCount) {
        scheduleWarmup(0);
    }
}

function processPendingThreshold() {
    thresholdTimer = null;
    const message = pendingThreshold;
    pendingThreshold = null;
    if (!message) {
        return;
    }
    const start = performance.now();
    try {
        const threshold = Number(message.threshold);
        const split = splitForThreshold(threshold);
        const options = {
            simplifyTolerance: Number.isFinite(message.simplifyTolerance) && message.simplifyTolerance > 0
                ? message.simplifyTolerance : 0,
            minArea: Number.isFinite(message.minArea) && message.minArea > 0 ? message.minArea : 0,
            // Gap closing is a geometry-build option supplied with `sources`.
            gapTolerance: sourceOptions.gapTolerance,
            fillColor: message.fillColor || null,
            lineColor: message.lineColor || null,
            lineWidth: Number.isFinite(message.lineWidth) ? message.lineWidth : 1,
            opacity: Number.isFinite(message.opacity) ? Math.max(0, Math.min(1, message.opacity)) : 0.3
        };
        const contours = contoursAt(split, options);
        self.postMessage({
            type: 'contours',
            requestId: message.requestId,
            threshold,
            split,
            final: Boolean(message.final),
            elapsedMs: performance.now() - start,
            count: polygons.length,
            contours
        });
        scheduleWarmup(split);
    } catch (error) {
        self.postMessage({type: 'error', requestId: message.requestId,
            error: String(error && error.message || error)});
    }
    // If slider messages arrived while this union was running, only the newest one remains.
    if (pendingThreshold && thresholdTimer === null) {
        thresholdTimer = setTimeout(processPendingThreshold, 0);
    }
}

self.onmessage = (event) => {
    const message = event.data || {};
    if (message.type === 'sources') {
        try {
            loadSources(message);
        } catch (error) {
            self.postMessage({type: 'error', requestId: message.requestId,
                error: String(error && error.message || error)});
        }
    } else if (message.type === 'threshold') {
        if (warmTimer !== null) {
            clearTimeout(warmTimer);
            warmTimer = null;
        }
        // Replace queued work; setTimeout allows a burst of slider events to coalesce.
        pendingThreshold = message;
        if (thresholdTimer === null) {
            thresholdTimer = setTimeout(processPendingThreshold, 0);
        }
    }
};
