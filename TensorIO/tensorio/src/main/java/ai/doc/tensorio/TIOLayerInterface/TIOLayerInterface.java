/*
 * TIOLayerInterface.java
 * TensorIO
 *
 * Created by Philip Dow on 7/6/2020
 * Copyright (c) 2020 - Present doc.ai (http://doc.ai)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
     * The type of layer represented by an interface
     */

    public enum Type {
        PixelBuffer,
        Vector
    }

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
     * values to a tensor buffer when a model is run on an multiple inputs or to associate an
     * output with a given name.
     */

    private String name;

    /**
     * The layers type, one of pixel buffer or vector (list)
     */

    private Type type;

    /**
     * The layer's mode, one of input, output, or placeholder.
     */

    private Mode mode;

    /**
     * The underlying data description.
     */

    private TIOLayerDescription dataDescription;

    /**
     * Initializes a @see TIOLayerInterface with a pixel buffer layer description.
     * @param name The name of the layer
     * @param mode The mode of the layer
     * @param description A vector layer description
     */

    public TIOLayerInterface(String name, Mode mode, TIOPixelBufferLayerDescription description) {
        this.name = name;
        this.mode = mode;
        this.dataDescription = description;
        this.type = Type.PixelBuffer;
    }

    /**
     * Initializes a @see TIOLayerInterface with a vector layer description.
     * @param name The name of the layer
     * @param mode The mode of the layer
     * @param description A vector layer description
     */

    public TIOLayerInterface(String name, Mode mode, TIOVectorLayerDescription description) {
        this.name = name;
        this.mode = mode;
        this.dataDescription = description;
        this.type = Type.Vector;
    }

    //region Getters and Setters

    public String getName() {
        return name;
    }

    public Type getType() {
        return type;
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