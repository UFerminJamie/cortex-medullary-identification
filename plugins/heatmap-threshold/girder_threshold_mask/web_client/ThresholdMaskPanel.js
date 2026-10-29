import View from '@girder/core/views/View';
import $ from 'jquery';
import {restRequest} from '@girder/core/rest';
import AnnotationModel from '@girder/large_image_annotation/models/AnnotationModel';
import router from '@girder/histomicsui/router';
import events from '@girder/histomicsui/events';

import mask from './logic';
import panelTemplate from './panel.pug';
import './panel.styl';

const classes = ['cortex', 'medulla'];

const ThresholdMaskPanel = View.extend({
    events: {
        'change .h-mask-source': '_sourceChanged',
        'input .h-heatmap-threshold': '_thresholdChanged',
        'change .h-heatmap-threshold': '_thresholdReleased',
        'input .h-heatmap-threshold-number': '_thresholdNumberChanged',
        'change .h-heatmap-threshold-number': '_thresholdNumberReleased',
        'click .h-heatmap-reset': '_resetThreshold',
        'input .h-mask-opacity': '_opacityChanged',
        'input .h-contour-line-width': '_lineWidthChanged',
        'click .h-mask-save': '_save',
        'click .h-threshold-close': 'close'
    },

    initialize(settings) {
        this.imageView = settings.imageView;
        this.collection = this.imageView.annotations;
        this.itemId = null;
        this.detectEnabled = true;
        this.viewer = null;
        this.active = false;
        this.threshold = 0;
        this.opacity = 0.5;
        this.lineWidth = 2;
        this.sources = {cortex: '', medulla: ''};
        this.manualSources = {cortex: false, medulla: false};
        this.status = '';
        this.canSave = false;
        this.saving = false;
        this.previews = {};
        this.sourcePromises = new Map();
        this.sourceReady = new Set();
        this.sourceWaiters = [];
        this.generation = 0;
        this.worker = null;
        this.workerReady = false;
        this.workerBusy = false;
        this.requestId = 0;
        this.pendingThreshold = null;
        this.pendingFinal = false;
        this.contourTimer = null;
        this.contours = null;
        this.contourSplit = null;
        this.contourTolerance = null;
        this.contourStats = null;
        this.sortedScores = null;
        this.sourceKey = '';
        this.saveWhenReady = false;
        this.listenTo(this.collection, 'sync update reset remove', this.render);
    },

    _availableSources() {
        return this.collection.filter((model) => {
            const annotation = model.get('annotation') || {};
            return !/^threshold (?:-?\d+(?:\.\d+)? )?mask /i.test(annotation.name || '') &&
                (/heatmap/i.test(annotation.name || '') ||
                /heatmap/i.test((annotation.attributes || {}).layer || ''));
        });
    },

    _detectSources(models) {
        if (!models.length) {
            return;
        }
        const selected = mask.detectRefinedSources(models.map((model) => ({
            id: model.id,
            name: (model.get('annotation') || {}).name || ''
        })));
        const changed = classes.filter((kind) => !this.manualSources[kind] &&
            selected[kind] && selected[kind] !== this.sources[kind]);
        changed.forEach((kind) => { this.sources[kind] = selected[kind]; });
        [...new Set(changed.map((kind) => selected[kind]))].forEach((id) => this._loadSource(id));
        if (changed.length) {
            this._setStatus('Refined heatmaps selected automatically. You can change either source.');
        }
    },

    render() {
        const models = this._availableSources();
        if (this.itemId && this.detectEnabled) {
            this._detectSources(models);
        }
        this.$el.html(panelTemplate({
            sources: models,
            maskSources: this.sources,
            threshold: this.threshold,
            opacity: this.opacity,
            lineWidth: this.lineWidth,
            status: this.status,
            saving: this.saving,
            canSave: this.canSave && this._hasDisplayContours()
        }));
        return this;
    },

    attach() {
        const host = this.imageView.$('.h-control-panel-container');
        if (!host.length) {
            return;
        }
        let panel = host.find('#h-thresholdmask-panel');
        if (!panel.length) {
            panel = $('<div id="h-thresholdmask-panel" class="s-panel h-control-panel hidden"/>').appendTo(host);
        }
        this.setElement(panel).render();
        this.$el.toggleClass('hidden', !this.active);
    },

    setItem(itemId, deferDetection = false) {
        if (this.itemId === itemId) {
            if (this.detectEnabled === deferDetection) {
                this.detectEnabled = !deferDetection;
                this.render();
            }
            return;
        }
        this._clearAll();
        this.itemId = itemId;
        this.detectEnabled = !deferDetection;
        this.active = false;
        this.threshold = 0;
        this.lineWidth = 2;
        this.sources = {cortex: '', medulla: ''};
        this.manualSources = {cortex: false, medulla: false};
        this.status = '';
        this.canSave = false;
        this.contours = null;
        this.contourSplit = null;
        this.contourTolerance = null;
        this.contourStats = null;
        this.sortedScores = null;
        this.sourceKey = '';
        this.saveWhenReady = false;
        if (itemId) {
            const folderId = this.imageView.model.get('folderId');
            if (folderId) {
                const generation = this.generation;
                restRequest({url: `annotation/folder/${folderId}/create`, error: null})
                    .done((access) => {
                        if (generation === this.generation) {
                            this.canSave = !!access;
                            this.render();
                        }
                    });
            }
        }
        this.render();
    },

    setViewer(viewer) {
        if (this.viewer !== viewer) {
            this._clearPreviews();
            this.viewer = viewer;
        }
        this._drawContours();
        this._buildPreviews();
    },

    viewerReady() {
        this._buildPreviews();
    },

    open() {
        if (!this.itemId) {
            return;
        }
        this.active = true;
        router.setQuery('analysis', null, {trigger: false});
        events.trigger('query:analysis', null);
        this.imageView.controlPanel.$el.addClass('hidden');
        this.attach();
        [...new Set(Object.values(this.sources).filter(Boolean))].forEach((id) => this._loadSource(id));
        this._buildPreviews();
    },

    close() {
        this.active = false;
        this._clearPreviews();
        if (this.contourTimer !== null) {
            window.clearTimeout(this.contourTimer);
            this.contourTimer = null;
        }
        if (this.worker) {
            this.worker.terminate();
            this.worker = null;
        }
        this.workerReady = false;
        this.workerBusy = false;
        this.pendingThreshold = null;
        this.pendingFinal = false;
        this.sourceKey = '';
        this.saveWhenReady = false;
        this.$el.addClass('hidden');
    },

    dispose() {
        this._clearAll();
        this.remove();
    },

    _setStatus(message) {
        this.status = message;
        this.$('.h-mask-status').text(message);
    },

    _removePreview(kind) {
        const preview = this.previews[kind];
        if (preview && this.viewer) {
            this.viewer.removeAnnotation(preview);
        }
        delete this.previews[kind];
    },

    _clearPreviews() {
        classes.forEach((kind) => this._removePreview(kind));
    },

    _clearAll() {
        this.generation += 1;
        this._clearPreviews();
        if (this.worker) {
            this.worker.terminate();
            this.worker = null;
        }
        this.workerReady = false;
        this.workerBusy = false;
        this.pendingThreshold = null;
        this.pendingFinal = false;
        if (this.contourTimer !== null) {
            window.clearTimeout(this.contourTimer);
            this.contourTimer = null;
        }
        this.sourceWaiters.forEach(({source, fetched, failed, reject}) => {
            source.off('g:fetched', fetched);
            source.off('g:error', failed);
            reject(new Error('Slide changed'));
        });
        this.sourceWaiters = [];
        this.sourcePromises.clear();
        this.sourceReady.clear();
        this.contours = null;
        this.contourSplit = null;
        this.contourTolerance = null;
        this.contourStats = null;
        this.sourceKey = '';
        this.sortedScores = null;
        this.saveWhenReady = false;
    },

    _sourceChanged(evt) {
        const kind = evt.currentTarget.dataset.kind;
        this.manualSources[kind] = true;
        this.sources[kind] = evt.currentTarget.value;
        this._clearPreviews();
        this.sourceKey = '';
        this.contours = null;
        this.contourSplit = null;
        this.contourTolerance = null;
        this.contourStats = null;
        this.sortedScores = null;
        this.saveWhenReady = false;
        if (this.worker) {
            this.worker.terminate();
            this.worker = null;
        }
        this.workerReady = false;
        this.workerBusy = false;
        this.pendingThreshold = null;
        this.pendingFinal = false;
        if (this.contourTimer !== null) {
            window.clearTimeout(this.contourTimer);
            this.contourTimer = null;
        }
        this._setStatus('');
        if (this.sources[kind]) {
            this._loadSource(this.sources[kind]);
        }
        this._buildPreviews();
    },

    _loadSource(id) {
        const source = this.collection.get(id);
        if (!source || this.sourceReady.has(id)) {
            this._buildPreviews();
            return;
        }
        const generation = this.generation;
        let pending = this.sourcePromises.get(id);
        if (!pending) {
            const annotation = source.get('annotation') || {};
            if ((annotation.elements || []).length && source._pageElements === false) {
                pending = Promise.resolve(source);
            } else if (source.get('loading') || source._inFetch) {
                pending = new Promise((resolve, reject) => {
                    const cleanup = () => {
                        source.off('g:fetched', fetched);
                        source.off('g:error', failed);
                        this.sourceWaiters = this.sourceWaiters.filter((waiter) => waiter.fetched !== fetched);
                    };
                    const fetched = () => { cleanup(); resolve(source); };
                    const failed = (error) => { cleanup(); reject(error); };
                    this.sourceWaiters.push({source, fetched, failed, reject});
                    source.on('g:fetched', fetched);
                    source.on('g:error', failed);
                }).then(() => {
                    if (source._pageElements === false) {
                        return source;
                    }
                    source._pageElements = false;
                    source._region = {};
                    return Promise.resolve(source.fetch()).then(() => source);
                });
            } else {
                source._pageElements = false;
                source._region = {};
                pending = Promise.resolve(source.fetch()).then(() => source);
            }
            this.sourcePromises.set(id, pending);
        }
        this._setStatus('Loading selected heatmap regions…');
        pending.then((model) => {
            if (generation !== this.generation || !Object.values(this.sources).includes(id)) {
                return;
            }
            const elements = ((model.get('annotation') || {}).elements || []);
            if (!elements.some((element) => Number.isFinite((element.user || {}).decision_score))) {
                this._setStatus('The selected heatmap has no numeric decision_score values.');
                return;
            }
            this.sourceReady.add(id);
            this._buildPreviews();
        }).catch(() => {
            if (generation === this.generation) {
                this.sourcePromises.delete(id);
                this._setStatus('Could not load a selected heatmap.');
            }
        });
    },

    _buildPreviews() {
        if (!this.active || !this.viewer || !this.viewer.viewer || !this.itemId) {
            return;
        }
        const ids = [...new Set(Object.values(this.sources).filter(Boolean))];
        if (!ids.length || !ids.every((id) => this.sourceReady.has(id))) {
            return;
        }
        const key = ids.slice().sort().join(':');
        if (this.sourceKey === key && (this.worker || this.workerReady)) {
            return;
        }
        const sources = ids.map((id) => {
            const source = this.collection.get(id);
            return source && {id, elements: ((source.get('annotation') || {}).elements || [])};
        });
        const elements = mask.collectSourceElements(sources);
        if (!elements.length) {
            this._setStatus('The selected heatmaps have no numeric scores.');
            return;
        }
        this._clearPreviews();
        this.sourceKey = key;
        this.contours = null;
        this.contourSplit = null;
        this.contourTolerance = null;
        this.contourStats = null;
        this._ensureWorker();
        this.workerReady = false;
        this.workerBusy = false;
        this._setStatus(`Sorting ${elements.length.toLocaleString()} scored regions in the browser…`);
        this.worker.postMessage({type: 'sources', elements});
    },

    _ensureWorker() {
        if (this.worker) {
            this.worker.terminate();
        }
        this.worker = new Worker('/static/built/plugins/threshold_mask/contour-worker.js');
        this.worker.onmessage = (evt) => this._workerMessage(evt.data || {});
        this.worker.onerror = (evt) => this._setStatus(`Contour worker error: ${evt.message || 'unknown error'}`);
    },

    _workerMessage(message) {
        if (message.type === 'ready') {
            this.workerReady = true;
            this.sortedScores = new Float64Array(message.scores);
            this._setStatus(`${message.count.toLocaleString()} scored regions loaded. Building full-detail contours…`);
            this._schedule(true);
            return;
        }
        if (message.type === 'error') {
            this.workerBusy = false;
            this._setStatus(`Could not build contours: ${message.error}`);
            return;
        }
        if (message.type !== 'contours' || message.requestId !== this.requestId) {
            return;
        }
        this.workerBusy = false;
        if (message.split === this._splitForThreshold(this.threshold)) {
            this.contours = message.contours;
            this.contourSplit = message.split;
            this.contourTolerance = message.final ? 0 : 8;
            this.contourStats = {count: message.count, split: message.split};
            this._drawContours();
            this.$('.h-mask-save').prop('disabled', !this.canSave || !this._hasDisplayContours() || this.saving);
            this._setStatus(`${message.contours.cortex.length} cortex and ${message.contours.medulla.length} medulla components; ${message.elapsedMs.toFixed(0)} ms. Fill-only preview. Save to create contour annotations.`);
        }
        if (this.pendingThreshold !== null && (
            this._splitForThreshold(this.pendingThreshold) !== message.split ||
            (this.pendingFinal && this.contourTolerance !== 0)
        )) {
            this._queueThreshold();
        } else {
            this.pendingThreshold = null;
            this.pendingFinal = false;
        }
        if (this.saveWhenReady && this._hasCurrentContours()) {
            this.saveWhenReady = false;
            this._save();
        }
    },

    _splitForThreshold(threshold) {
        if (!this.sortedScores) {
            return null;
        }
        let low = 0;
        let high = this.sortedScores.length;
        while (low < high) {
            const middle = (low + high) >>> 1;
            if (this.sortedScores[middle] >= threshold) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        return low;
    },

    _hasDisplayContours() {
        return Boolean(this.contours && this.contourSplit === this._splitForThreshold(this.threshold) &&
            (this.contours.cortex.length || this.contours.medulla.length));
    },

    _hasCurrentContours() {
        return Boolean(this._hasDisplayContours() && this.contourTolerance === 0);
    },

    _thresholdChanged(evt) {
        this.threshold = Number(evt.currentTarget.value);
        this.$('.h-heatmap-threshold-number').val(this.threshold.toFixed(1));
        this.$('.h-heatmap-threshold-value').text(this.threshold.toFixed(1));
        this._schedule(false);
    },

    _thresholdReleased() {
        this._schedule(true);
    },

    _thresholdNumberChanged(evt) {
        const raw = evt.currentTarget.value.trim();
        const parsed = raw === '' ? this.threshold : Number(raw);
        this.threshold = Number.isFinite(parsed)
            ? Math.round(Math.max(-5, Math.min(5, parsed)) * 10) / 10
            : this.threshold;
        this.$('.h-heatmap-threshold').val(String(this.threshold));
        this.$('.h-heatmap-threshold-number').val(this.threshold.toFixed(1));
        this.$('.h-heatmap-threshold-value').text(this.threshold.toFixed(1));
        this._schedule(false);
    },

    _thresholdNumberReleased() {
        this._schedule(true);
    },

    _resetThreshold() {
        this.threshold = 0;
        this.$('.h-heatmap-threshold').val('0');
        this.$('.h-heatmap-threshold-number').val('0.0');
        this.$('.h-heatmap-threshold-value').text('0.0');
        this._schedule(true);
    },

    _opacityChanged(evt) {
        this.opacity = Number(evt.currentTarget.value);
        this.$('.h-mask-opacity-value').text(`${(this.opacity * 100).toFixed(0)}%`);
        this._drawContours();
    },

    _lineWidthChanged(evt) {
        this.lineWidth = Number(evt.currentTarget.value);
        this.$('.h-contour-line-width-value').text(`${this.lineWidth}px`);
    },

    _schedule(final = false) {
        const split = this._splitForThreshold(this.threshold);
        const wantedTolerance = final ? 0 : 8;
        if (split !== null && this.contours && this.contourSplit === split &&
            (this.contourTolerance === 0 || this.contourTolerance === wantedTolerance)) {
            this._drawContours();
            this.$('.h-mask-save').prop('disabled', !this.canSave || !this._hasDisplayContours() || this.saving);
            this.pendingThreshold = null;
            this.pendingFinal = false;
            return;
        }
        this.$('.h-mask-save').prop('disabled', true);
        this.pendingThreshold = this.threshold;
        this.pendingFinal = this.pendingFinal || final;
        if (!this.active || !this.workerReady || this.workerBusy) {
            return;
        }
        if (this.contourTimer !== null) {
            window.clearTimeout(this.contourTimer);
        }
        this.contourTimer = window.setTimeout(() => {
            this.contourTimer = null;
            if (!this.workerBusy && this.pendingThreshold !== null) {
                this._sendThreshold(this.pendingThreshold, this.pendingFinal);
            }
        }, 120);
    },

    _queueThreshold() {
        if (this.pendingThreshold === null) {
            return;
        }
        this._schedule(this.pendingFinal);
    },

    _sendThreshold(threshold, final = false) {
        if (!this.worker || !this.workerReady) {
            return;
        }
        this.workerBusy = true;
        this.pendingThreshold = null;
        this.pendingFinal = false;
        this.$('.h-mask-save').prop('disabled', true);
        this._setStatus('Updating merged contour regions locally…');
        this.worker.postMessage({
            type: 'threshold', requestId: ++this.requestId, threshold, final,
            simplifyTolerance: final ? 0 : 8,
            opacity: this.opacity,
            lineWidth: this.lineWidth
        });
    },

    _drawContours() {
        if (!this._hasDisplayContours() || !this.viewer) {
            return;
        }
        this._clearPreviews();
        classes.forEach((kind) => {
            const rgb = kind === 'medulla' ? '255, 0, 0' : '0, 0, 255';
            const elements = (this.contours[kind] || []).map((element, index) => Object.assign({}, element, {
                id: `threshold-${kind}-${index}`,
                fillColor: `rgba(${rgb}, ${this.opacity})`,
                lineColor: 'rgba(0, 0, 0, 0)',
                lineWidth: 0.5
            }));
            if (!elements.length) {
                return;
            }
            const preview = new AnnotationModel({
                _id: `threshold-contour-preview-${kind}-${this.itemId}`,
                itemId: this.itemId,
                annotation: {name: `threshold ${kind} fill preview`, elements}
            });
            this.previews[kind] = preview;
            this.viewer.drawAnnotation(preview, {fetch: false});
        });
    },

    _save() {
        if (this.saving || !this.canSave || !this._hasDisplayContours()) {
            return;
        }
        if (!this._hasCurrentContours()) {
            this.saveWhenReady = true;
            this.$('.h-mask-save').prop('disabled', true);
            this._setStatus('Finishing full-detail contours; they will save automatically when ready…');
            this._schedule(true);
            return;
        }
        const ids = [...new Set(Object.values(this.sources).filter(Boolean))];
        const sourceAnnotations = ids.map((id) => {
            const source = this.collection.get(id);
            return {id, name: source ? ((source.get('annotation') || {}).name || '') : ''};
        });
        const threshold = this.threshold;
        const thresholdText = mask.formatThreshold(threshold);
        const stats = this.contourStats || {count: null, split: null};
        const models = classes.filter((kind) => (this.contours[kind] || []).length).map((kind) => {
            const rgb = kind === 'medulla' ? '255, 0, 0' : '0, 0, 255';
            const elements = this.contours[kind].map((element) => Object.assign({}, element, {
                fillColor: `rgba(${rgb}, ${this.opacity})`,
                lineColor: `rgb(${rgb})`,
                lineWidth: this.lineWidth
            }));
            return new AnnotationModel({
                itemId: this.itemId,
                annotation: {
                    name: `threshold ${thresholdText} contour ${kind}`,
                    attributes: {thresholdContour: {
                        generatedBy: 'cortex-medulla-threshold-v1',
                        algorithmVersion: '0.2.0',
                        class: kind,
                        threshold,
                        thresholdRule: 'medulla score >= threshold; cortex score < threshold',
                        sourceAnnotationIds: ids,
                        sourceAnnotations,
                        scoredRegionCount: stats.count,
                        medullaRegionCount: stats.split,
                        cortexRegionCount: stats.count === null ? null : stats.count - stats.split,
                        coordinateSystem: 'slide-level-0',
                        opacity: this.opacity,
                        lineWidth: this.lineWidth
                    }},
                    elements
                }
            });
        });
        if (!models.length) {
            this._setStatus('No contour components are available at this threshold.');
            return;
        }
        const generation = this.generation;
        this.saving = true;
        this.$('.h-mask-save').prop('disabled', true);
        this._setStatus(`Saving ${models.length} contour annotation${models.length === 1 ? '' : 's'}…`);
        Promise.all(models.map((model) => model.save())).then(() => {
            if (generation !== this.generation) {
                return;
            }
            models.forEach((model) => {
                model.set('displayed', true);
                this.collection.add(model);
            });
            this._clearPreviews();
            this._setStatus(`Saved ${models.map((model) => model.get('annotation').name).join(' and ')}.`);
        }).catch(() => {
            if (generation === this.generation) {
                this._setStatus('Could not save all contour annotations. Check the annotation list before retrying.');
            }
        }).finally(() => {
            this.saving = false;
            this.$('.h-mask-save').prop('disabled', !this.canSave || !this._hasDisplayContours());
        });
    }
});

export default ThresholdMaskPanel;
