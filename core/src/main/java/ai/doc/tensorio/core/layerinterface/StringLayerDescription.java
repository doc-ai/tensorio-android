/*
 * StringLayerDescription.java
 * TensorIO
 *
 * Created by Philip Dow on 7/27/2020
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

import static java.lang.Math.abs;

/**
 * The description of a string (raw bytes) input or output layer.
 *
 * String inputs and outputs capture the raw bytes passed to a tensor or
 * extracted from one. Both TensorFlow and Caffe call this data type a "string",
 * although here it will be captured in a BytBuffer object wrapping the underlying
 * bytes.
 *
 * No quantization or dequantization is applied to raw bytes. They are copied
 * directly to the underling tensor.
 *
 * If your model requests string (byte) inputs or outputs and you need to work with the ByteBuffer
 * in a thread off the one that inference is performed on, you should send a copy of the buffer
 * into the model or make a copy of the one returned before doing additional work, as the ByteBuffer
 * used by the model may be cached and re-used for the next step of inference.
 */

public class StringLayerDescription extends LayerDescription {

    /**
     * The shape of the underlying layer.
     */

    private final int[] shape;

    /**
     * The length of the vector in terms of its number of elements.
     */

    private int length;

    /**
     * Designated initializer. Creates a string description from the properties parsed in a model.json
     * file.
     *
     * @param shape The shape of the underlying tensor
     * @param batched True if the layer supports batched execution, false otherwise
     * @param dtype The type of data this layer expects or produces
     * @return instancetype A read-only instance of `StringLayerDescription`
     */

    public StringLayerDescription(int[] shape, boolean batched, DataType dtype) {
        this.shape = shape;
        this.batched = batched;
        this.dtype = dtype;

        this.length = 1;
        for (int i : shape) {
            this.length *= abs(i);
        }
    }

    public int[] getShape() {
        return shape;
    }

    public int getLength() {
        return length;
    }

    @Override
    public int[] getTensorShape() {
        return getShape();
    }

}
