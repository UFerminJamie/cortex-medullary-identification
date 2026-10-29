const assert = require('node:assert/strict');
const {test} = require('node:test');

const logic = require('../girder_threshold_mask/web_client/logic');

test('auto-detects refined cortex and medulla heatmaps independently', () => {
    assert.deepEqual(logic.detectRefinedSources([
        {id: 'old', name: 'cortical heatmap'},
        {id: 'medulla', name: 'Medullary Refined Heatmap'},
        {id: 'cortex', name: 'Cortical Refined Heatmap'}
    ]), {cortex: 'cortex', medulla: 'medulla'});
    assert.deepEqual(logic.detectRefinedSources([
        {id: 'cortex', name: 'Cortical Refined Heatmap'}
    ]), {cortex: 'cortex', medulla: ''});
});

test('keeps only scored geometry and deduplicates a source selected in both roles', () => {
    const scored = {type: 'rectangle', user: {decision_score: 0.25}};
    const unscored = {type: 'rectangle', user: {decision_score: '0.4'}};
    assert.deepEqual(logic.collectSourceElements([
        {id: 'same', elements: [scored, unscored]},
        {id: 'same', elements: [scored]}
    ]), [scored]);
});

test('formats threshold values consistently in saved contour names', () => {
    assert.equal(logic.formatThreshold(0), '0');
    assert.equal(logic.formatThreshold(-1.2), '-1.2');
    assert.equal(logic.formatThreshold(2.3000001), '2.3');
});
