package ai.doc.tensorio.TIOLayerInterface;

/**
 * Encapsulates information about the input, output, and placeholder layers of a model, fully described by a
 * `TIOLayerDescription`. Used internally by a model when parsing its description. Also used to
 * match inputs, outputs, and placeholders to their corresponding layers.
 *
 * This is an algebraic data type inspired by Remodel: https://github.com/facebook/remodel.
 * In Swift it would be an Enumeration with Associated Values. The intent is to capture the
 * variety of inputs and outputs a model can accept and produce in a unified interface.
 *
 * Normally you will not need to interact with this class, although you may request a
 * `TIOLayerDescription` from a conforming `TIOModel` for inputs or outputs that you are specifically
 * interested in, for example, a pixel buffer input when you want greater control over scaling
 * and clipping an image before passing it to the model.
 */

public class TIOLayerInterface {

    /**
     * Layers Mode is one of input, output, or placeholder.
     */

    public enum Mode {
        Input,
        Output,
        Placeholder
    }

    /**
     * The name of the model interface
     *
     * May corresponding to an actual layer name or be your own name. The name will be used to copy
     * values to a tensor buffer when a model is run on an `NSDictionary` input or to associate an
     * output with a given name.
     */

    private String name;

    /**
     * The layer's mode, one of input, output, or placeholder.
     */

    private Mode mode;

    /**
     * The underlying data description.
     */

    private TIOLayerDescription dataDescription;

    /**
     * Initializes a @see TIOLayerInterface with a layer description.
     *
     * @param description Description of the expected  buffer
     */

    public TIOLayerInterface(String name, Mode mode, TIOLayerDescription description) {
        this.name = name;
        this.mode = mode;
        this.dataDescription = description;
    }

    //region Getters and Setters

    public String getName() {
        return name;
    }

    public Mode getMode() {
        return mode;
    }

    public TIOLayerDescription getDataDescription() {
        return dataDescription;
    }

    //endregion

    /**
     * Initializes a @see TIOLayerInterface with a vector description, e.g. the description of a vector,
     * matrix, or other tensor.
     *
     * @param vectorDescription Description of the expected vector
     */

    /*
    public TIOLayerInterface(String name, boolean isInput, TIOVectorLayerDescription vectorDescription) {
        this.name = name;
        this.input = isInput;
        this.dataDescription = vectorDescription;
    }
    */

    /**
     * Use this function to switch on the underlying description.
     * <p>
     * When preparing inputs and capturing outputs, a `TIOModel` uses the underlying description of a layer
     * in order to determine how to move bytes around.
     */
        /*
        -(void)matchCasePixelBuffer:(TIOPixelBufferMatcher)pixelBufferMatcher caseVector:(TIOVectorMatcher)vectorMatcher;
        }
        */
}