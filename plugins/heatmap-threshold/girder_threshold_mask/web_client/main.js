// Girder's HTML lists plugin scripts by discovery order, which can put this
// bundle before HistomicsUI and large_image_annotation.  The app-start event
// fires after all scripts have loaded but before the views are instantiated.
function boot() {
const {wrap} = require('@girder/core/utilities/PluginUtils');
const ImageView = require('@girder/histomicsui/views/body/ImageView').default;
const HeaderAnalysesView = require('@girder/histomicsui/views/layout/HeaderAnalysesView').default;
const events = require('@girder/histomicsui/events').default;
const ThresholdMaskPanel = require('./ThresholdMaskPanel').default;

function connectViewer(imageView, viewer) {
    const panel = imageView._thresholdMaskPanel;
    if (!panel) {
        return;
    }
    panel.setViewer(viewer);
    if (viewer && imageView._thresholdMaskConnectedViewer !== viewer) {
        imageView._thresholdMaskConnectedViewer = viewer;
        imageView.listenTo(viewer, 'g:imageRendered', () => panel.viewerReady());
    }
}

wrap(ImageView, 'render', function (render, ...args) {
    const panel = this._thresholdMaskPanel;
    if (panel && panel.itemId !== this.model.id) {
        panel.setItem(this.model.id, true);
    }
    const result = render.apply(this, args);
    if (!this._thresholdMaskPanel) {
        this._thresholdMaskPanel = new ThresholdMaskPanel({imageView: this});
        this.listenTo(events, 'threshold-mask:open', () => this._thresholdMaskPanel.open());
        this.listenTo(events, 'query:analysis', (api) => {
            if (api) {
                this._thresholdMaskPanel.close();
            }
        });
        this.listenTo(this, 'h:viewerWidgetCreated', (viewer) => connectViewer(this, viewer));
        this.listenTo(this.annotationSelector, 'h:annotationFillOpacity', () => {
            this._thresholdMaskPanel.styleDirty = true;
            this._thresholdMaskPanel._schedule();
        });
    }
    this._thresholdMaskPanel.setItem(this.model.id);
    connectViewer(this, this.viewerWidget);
    this._thresholdMaskPanel.attach();
    return result;
});

wrap(ImageView, 'destroy', function (destroy, ...args) {
    if (this._thresholdMaskPanel) {
        this._thresholdMaskPanel.dispose();
        this._thresholdMaskPanel = null;
    }
    return destroy.apply(this, args);
});

function ensureAnalysisEntry(view) {
    if (!view.image || !view.el || !view.el.isConnected) {
        return;
    }
    if (view._thresholdMaskUpdatingMenu) {
        return;
    }
    view._thresholdMaskUpdatingMenu = true;
    try {
        let menu = view.$('.h-analyses-dropdown');
        if (!menu.length) {
            view.$el.html('<a class="h-analyses-dropdown-link" data-toggle="dropdown"><span class="icon-tasks"></span> Analyses <span class="icon-down-open"></span></a><ul class="h-analyses-dropdown dropdown-menu" role="menu"></ul>');
            menu = view.$('.h-analyses-dropdown');
        }
        if (!menu.find('.h-local-threshold-analysis').length) {
            menu.prepend('<li><a class="h-local-threshold-analysis" tabindex="0" href="#">cortex-medulla-threshold</a></li>');
        }
        view.$el.removeClass('hidden');
    } finally {
        view._thresholdMaskUpdatingMenu = false;
    }
}

wrap(HeaderAnalysesView, 'render', function (render, ...args) {
    const result = render.apply(this, args);
    if (!this._thresholdMaskMenuObserver) {
        this._thresholdMaskMenuObserver = new MutationObserver(() => ensureAnalysisEntry(this));
        this._thresholdMaskMenuObserver.observe(this.el, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['class']
        });
        this.$el.on('click.thresholdMask', '.h-local-threshold-analysis', (evt) => {
            evt.preventDefault();
            events.trigger('threshold-mask:open');
        });
    }
    ensureAnalysisEntry(this);
    return result;
});

wrap(HeaderAnalysesView, 'destroy', function (destroy, ...args) {
    if (this._thresholdMaskMenuObserver) {
        this._thresholdMaskMenuObserver.disconnect();
        this._thresholdMaskMenuObserver = null;
    }
    this.$el.off('.thresholdMask');
    return destroy.apply(this, args);
});
}

window.girder.events.once('g:appload.before', boot);
