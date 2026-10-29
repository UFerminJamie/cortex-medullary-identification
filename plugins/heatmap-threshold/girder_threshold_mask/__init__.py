from girder.plugin import GirderPlugin


class ThresholdMaskPlugin(GirderPlugin):
    DISPLAY_NAME = 'cortex-medulla-threshold'
    CLIENT_SOURCE_PATH = 'web_client'

    def load(self, info):
        # Thresholding runs entirely in the browser; there are no REST routes.
        pass
