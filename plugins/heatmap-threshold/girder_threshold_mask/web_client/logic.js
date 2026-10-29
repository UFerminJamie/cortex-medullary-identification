function formatThreshold(threshold) {
    return String(Number(threshold.toFixed(1)));
}

function detectRefinedSources(sources) {
    const find = (kind) => {
        const className = kind === 'cortex' ? /cortical|cortex/i : /medullary|medulla/i;
        return sources.find((source) => /refined/i.test(source.name) &&
            /heatmap/i.test(source.name) && className.test(source.name));
    };
    const cortex = find('cortex');
    const medulla = find('medulla');
    return {cortex: cortex ? cortex.id : '', medulla: medulla ? medulla.id : ''};
}

function collectSourceElements(sources) {
    const seen = new Set();
    const elements = [];
    sources.forEach((source) => {
        if (!source || seen.has(source.id)) {
            return;
        }
        seen.add(source.id);
        (source.elements || []).forEach((element) => {
            if (Number.isFinite((element.user || {}).decision_score)) {
                elements.push(element);
            }
        });
    });
    return elements;
}

module.exports = {formatThreshold, detectRefinedSources, collectSourceElements};
