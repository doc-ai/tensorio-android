package ai.doc.tensorio.TIOModel;

public class TIOModelOptions {

    /**
     * Preferred device position.
     * <p>
     * If the device position is unspecified at initialization, `0` will be used,
     * which will then typically default to the back facing camera.
     */

    private String devicePosition;

    /**
     * Converts a string representation of a capture device position, e.g. a camera position,
     * to a Camera ID
     *
     * @param descriptor A string representation of a device position. 'front' and 'back' are
     *                   the only values currently supported.
     */
    public TIOModelOptions(String descriptor) {
        if (descriptor.equals("front")) {
            this.devicePosition = "1";
        } else {
            this.devicePosition = "0";
        }
    }

    public String getDevicePosition() {
        return devicePosition;
    }
}
