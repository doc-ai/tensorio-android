/*
 * LayerInterface.java
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

package ai.doc.tensorio.core.layerinterface;

import java.util.function.Consumer;

/**
 * Encapsulates information about the input, output, and placeholder layers of a model, fully described by a
 * `LayerDescription`. Used internally by a model when parsing its description. Also used to
 * match inputs, outputs, and placeholders to their corresponding layers.
 *
 * This is an algebraic data type inspired by Remodel: https://github.com/facebook/remodel.
 * In Swift it would be an Enumeration with Associated Values. The intent is to capture the
 * variety of inputs and outputs a model can accept and produce in a unified interface.
 *
 * Normally you will not need to interact with this class, although you may request a
 * `LayerDescription` from a conforming `Model` for inputs or outputs that you are specifically
 * interested in, for example, a pixel buffer input when you want greater control over scaling
 * and clipping an image before passing it to the model.
 */

public class LayerInterface {

    /**
     * The type of layer represented by an interface
     */

    public enum Type {
        PixelBuffer,
        Vector,
        String
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

    private LayerDescription layerDescription;

    /**
     * Initializes a @see LayerInterface with a pixel buffer layer description.
     *
     * @param name The name of the layer
     * @param mode The mode of the layer
     * @param description A vector layer description
     */

    public LayerInterface(String name, Mode mode, PixelBufferLayerDescription description) {
        this.name = name;
        this.mode = mode;
        this.layerDescription = description;
        this.type = Type.PixelBuffer;
    }

    /**
     * Initializes a @see LayerInterface with a vector layer description.
     *
     * @param name The name of the layer
     * @param mode The mode of the layer
     * @param description A vector layer description
     */

    public LayerInterface(String name, Mode mode, VectorLayerDescription description) {
        this.name = name;
        this.mode = mode;
        this.layerDescription = description;
        this.type = Type.Vector;
    }

    /**
     * Initializes a @see LayerInterface with a string (bytes) layer description.
     *
     * @param name The name of the layer
     * @param mode The mode of the layer
     * @param description A string (bytes) layer description
     */

    public LayerInterface(String name, Mode mode, StringLayerDescription description) {
        this.name = name;
        this.mode = mode;
        this.layerDescription = description;
        this.type = Type.String;
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

    public int[] getTensorShape() {
        return layerDescription.getTensorShape();
    }

    //endregion

    /**
     * The primary interface to the underlying layer description. Rather than accessing the description
     * directly, call this function, which effectively switches on the type of the layer and calls
     * back to the lambdas you have provided for each type.
     *
     * For example:
     *
     * <code>
     *     layer.doCase((vectorLayer) -> {
     *         // your code
     *     }, (pixelLayer) -> {
     *         // your code
     *     }, (stringLayer) -> {
     *         // your code
     *     });
     * </code>
     *
     * @param vectorLayer A consuming lambda that takes the vector layer description as a parameter
     * @param pixelLayer A consuming lambda that takes the pixel layer description as a parameter
     * @param stringLayer A consuming lambda that takes the string (bytes) layer description as a parameter
     */

    public void doCase(
            Consumer<VectorLayerDescription> vectorLayer,
            Consumer<PixelBufferLayerDescription> pixelLayer,
            Consumer<StringLayerDescription> stringLayer) {
        switch (this.type) {
            case Vector:
                vectorLayer.accept((VectorLayerDescription)this.layerDescription);
                break;
            case PixelBuffer:
                pixelLayer.accept((PixelBufferLayerDescription)this.layerDescription);
                break;
            case String:
                stringLayer.accept((StringLayerDescription)this.layerDescription);
                break;
        }
    }
}