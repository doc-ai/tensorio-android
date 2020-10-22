/*
 * Converter.java
 * TensorIO
 *
 * Created by Philip Dow on 10/14/2020
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

package ai.doc.tensorio.tensorflow.data;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.nio.ByteBuffer;

import ai.doc.tensorio.core.layerinterface.LayerDescription;

/**
 * TensorFlow models use ByteBuffers by way of Tensors to write data into models and read data out of them,
 * so conforming TensorFlow data converts must know how to work with ByteBuffers.
 */

public interface Converter {

    /**
     * Creates a ByteBuffer to hold data for input or output to a TensorFlow model using the parameters
     * in the layer description.
     *
     * @param description A description of the layer to create a byte buffer for
     * @param batchSize The number of items that will be written into this buffer
     * @return ByteBuffer ready to be filled with input or output data.
     */

    public ByteBuffer createBackingBuffer(@NonNull LayerDescription description, int batchSize);

    /**
     * Converts an array of objects (a column of data) to a ByteBuffer, used to prepare data for a writing into a model.
     *
     * TODO: Optimize
     * For now the column method allocates a byte buffer large enough to hold all the data and then calls
     * the object method for each item in the column, accumulating byte buffers for each item that it then
     * copies into the holding buffer. There is no caching and this is surely inefficient, but it will do
     * for a start.
     *
     * @param c           A column (array) of a number of types that can be converted into a ByteBuffer
     * @param description A description of the layer with instructions on how to make the conversion
     * @param cache       A pre-existing byte buffer to use, which will be returned if not null. If a cache
     *                    is provided it will be rewound before being used.
     * @return ByteBuffer ready for use with a TensorFlow model
     * @throws IllegalArgumentException Raised if the input object o is not of one of the supported
     *                                  types or is the wrong length
     */

    public ByteBuffer toByteBuffer(@NonNull Object[] c, @NonNull LayerDescription description, @Nullable ByteBuffer cache) throws IllegalArgumentException;

    /**
     * Converts an Object to a ByteBuffer, used to prepare data for a writing into a model.
     *
     * @param o           One of a number of types that can be converted into a ByteBuffer
     * @param description A description of the layer with instructions on how to make the conversion
     * @param cache       A pre-existing byte buffer to use, which will be returned if not null. If a cache
     *                    is provided it will be rewound before being used.
     * @return ByteBuffer ready for use with a TensorFlow model
     * @throws IllegalArgumentException Raised if the input object o is not of one of the supported
     *                                  types or is the wrong length
     */

    public ByteBuffer toByteBuffer(@NonNull Object o, @NonNull LayerDescription description, @Nullable ByteBuffer cache) throws IllegalArgumentException;

    /**
     * Converts a ByteBuffer to an object, used to read data from a model.
     *
     * @param buffer      A ByteBuffer read from a TensorFlow model
     * @param description A description of the layer with instructions on how to make the conversion
     * @return One of a number of native types such as an array of floats or a Bitmap
     */

    public Object fromByteBuffer(@NonNull ByteBuffer buffer, @NonNull LayerDescription description);
}
