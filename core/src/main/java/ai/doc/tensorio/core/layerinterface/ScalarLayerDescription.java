/*
 * ScalarLayerDescription.java
 * TensorIO
 *
 * Created by Philip Dow on 12/19/2021
 * Copyright (c) 2021 - Present doc.ai (http://doc.ai)
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

import ai.doc.tensorio.core.data.Dequantizer;
import ai.doc.tensorio.core.data.Quantizer;

import static java.lang.Math.abs;

/**
 * The description of a scalar (single valued) input or output later. Scalar inputs and outputs are
 * single valued without dimension.
 *
 * @warning
 * A `ScalarLayerDescription`'s length is different than the byte length of a `Data` object.
 * For example a quantized `Scalar` (uint8_t) of length 4 will occupy 4 bytes of memory but an
 * unquantized `Scalar` (float_t) of length 4 will occupy 16 bytes of memory.
 */

public class ScalarLayerDescription extends LayerDescription {

    /**
     * The shape of the underlying layer
     */

    private final int[] shape;

    /**
     * The length of the vector in terms of its number of elements.
     */

    private int length;

    /**
     * true if there are labels associated with this layer, false otherwise.
     */

    /**
     * A function that converts a vector from unquantized values to quantized values
     */

    private Quantizer quantizer;

    /**
     * A function that converts a vector from quantized values to unquantized values
     */

    private Dequantizer dequantizer;

    /**
     * Designated initializer. Creates a vector description from the properties parsed in a model.json
     * file.
     *
     * @param shape       The layer's shape which should be [-1], [1], or [-1,1]
     * @param batched     True if the layer supports batched execution, false otherwise
     * @param quantized   True if the values are quantized
     * @param quantizer   A function that transforms unquantized values to quantized input
     * @param dequantizer A function that transforms quantized output to unquantized values
     */

    public ScalarLayerDescription(int[] shape, boolean batched, boolean quantized, Quantizer quantizer, Dequantizer dequantizer, DataType dtype) {
        this.shape = shape;

        // Total Volume
        this.length = 1;
        for (int i : shape) {
            length *= abs(i);
        }

        this.batched = batched;
        this.quantized = quantized;
        this.quantizer = quantizer;
        this.dequantizer = dequantizer;
        this.dtype = dtype;
    }

    //region Getters and Setters

    public int[] getShape() {
        return shape;
    }

    public int getLength() {
        return length;
    }

    public Quantizer getQuantizer() {
        return quantizer;
    }

    public Dequantizer getDequantizer() {
        return dequantizer;
    }

    @Override
    public int[] getTensorShape() {
        return getShape();
    }

}
