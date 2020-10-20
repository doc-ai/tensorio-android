/*
 * StringConverter.java
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

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

import ai.doc.tensorio.core.layerinterface.DataType;
import ai.doc.tensorio.core.layerinterface.LayerDescription;
import ai.doc.tensorio.core.layerinterface.StringLayerDescription;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

public class StringConverter implements Converter {

    @Override
    public ByteBuffer createBackingBuffer(@NonNull LayerDescription description, int batchSize) {
        boolean quantized = ((StringLayerDescription)description).isQuantized();
        DataType dtype = ((StringLayerDescription)description).getDtype();
        int length = ((StringLayerDescription)description).getLength();

        // Compute buffer length

        int bufferLength = 0;

        switch (dtype) {
            case UInt8:
                bufferLength = length;
                break;
            case Int32:
                bufferLength = length * 4;
                break;
            case Int64:
                bufferLength = length * 8;
                break;
            case Float32:
                bufferLength = length * 4;
                break;
            case Unknown:;
                // Unspecified input is assumed float
                bufferLength = length * 4;
                break;
        }

        if (quantized) {
            // Override for a quantized layer
            bufferLength = length;
        }

        bufferLength *= batchSize;

        // Create buffer

        ByteBuffer buffer = ByteBuffer.allocateDirect(bufferLength);
        buffer.order(ByteOrder.nativeOrder());

        return buffer;
    }

    @Override
    public ByteBuffer toByteBuffer(@NonNull Object[] c, @NonNull LayerDescription description, @Nullable ByteBuffer cache) throws  IllegalArgumentException {
        ByteBuffer buffer = (cache != null) ? cache : createBackingBuffer(description, c.length);
        buffer.rewind();

        for (Object o : c) {
            buffer.put((ByteBuffer) toByteBuffer(o, description, null).rewind());
        }

        return buffer;
    }

    /**
     * Processes an input to a model and copies it into a possibly cached ByteBuffer for consumption
     * by the model. In this case, the provided bytes or ByteBuffer are copied directly into the
     * ByteBuffer that will be used by the model.
     *
     * @param o           One of a number of types that can be converted into a ByteBuffer
     * @param description A description of the layer with instructions on how to make the conversion
     * @param cache       A pre-existing byte buffer to use, which will be returned if not null. If a cache
     *                    is provided it will be rewound before being used.
     * @return The ByteBuffer that will be consumed by the model
     * @throws IllegalArgumentException if the input Object o is not of type [byte[]], [float[]],
     *                    [ByteBuffer], or [FloatBuffer]
     */

    @Override
    public ByteBuffer toByteBuffer(@NonNull Object o, @NonNull LayerDescription description, @Nullable ByteBuffer cache) throws IllegalArgumentException {
        DataType dtype = description.getDtype();

        // Barf

        if (o instanceof byte[]) {
            return toByteBuffer((byte[])o, description, cache);
        } else if (o instanceof float[]) {
            return toByteBuffer((float[])o, description, cache);
        } else if (o instanceof int[]) {
            return toByteBuffer((int[])o, description, cache);
        } else if (o instanceof long[]) {
            return toByteBuffer((long[])o, description, cache);
        } else if (o instanceof FloatBuffer) {
            return toByteBuffer((FloatBuffer)o, description, cache);
        } else if (o instanceof IntBuffer) {
            return toByteBuffer((IntBuffer)o, description, cache);
        } else if (o instanceof LongBuffer) {
            return toByteBuffer((LongBuffer) o, description, cache);
        } else if (o instanceof ByteBuffer) {
            ((ByteBuffer)o).rewind();
            switch (dtype) {
                case UInt8:
                    return toByteBuffer((ByteBuffer)o, description, cache);
                case Int32:
                    return toByteBuffer(((ByteBuffer)o).asIntBuffer(), description, cache);
                case Int64:
                    return toByteBuffer(((ByteBuffer)o).asLongBuffer(), description, cache);
                case Float32:
                case Unknown:
                    return toByteBuffer(((ByteBuffer)o).asFloatBuffer(), description, cache);
            }
        } else {
            throw BadInputException();
        }

        return null;
    }

    public ByteBuffer toByteBuffer(@NonNull byte[] bytes, @NonNull LayerDescription description, @Nullable ByteBuffer cache) throws IllegalArgumentException {
        // Create a buffer if no reusable cache is provided

        ByteBuffer buffer = (cache != null) ? cache : createBackingBuffer(description, 1);
        buffer.rewind();

        // Write the bytes

        buffer.put(bytes);

        return buffer;
    }

    public ByteBuffer toByteBuffer(@NonNull float[] floats, @NonNull LayerDescription description, @Nullable ByteBuffer cache) throws IllegalArgumentException {
        // Create a buffer if no reusable cache is provided

        ByteBuffer buffer = (cache != null) ? cache : createBackingBuffer(description, 1);
        buffer.rewind();

        // Write the floats

        buffer.asFloatBuffer().put(floats);

        return buffer;
    }

    public ByteBuffer toByteBuffer(@NonNull int[] ints, @NonNull LayerDescription description, @Nullable ByteBuffer cache) throws IllegalArgumentException {
        // Create a buffer if no reusable cache is provided

        ByteBuffer buffer = (cache != null) ? cache : createBackingBuffer(description, 1);
        buffer.rewind();

        // Write the ints

        buffer.asIntBuffer().put(ints);

        return buffer;
    }

    public ByteBuffer toByteBuffer(@NonNull long[] longs, @NonNull LayerDescription description, @Nullable ByteBuffer cache) throws IllegalArgumentException {
        // Create a buffer if no reusable cache is provided

        ByteBuffer buffer = (cache != null) ? cache : createBackingBuffer(description, 1);
        buffer.rewind();

        // Write the longs

        buffer.asLongBuffer().put(longs);

        return buffer;
    }

    public ByteBuffer toByteBuffer(@NonNull ByteBuffer byteBuffer, @NonNull LayerDescription description, @Nullable ByteBuffer cache) throws IllegalArgumentException {
        // Create a buffer if no reusable cache is provided

        ByteBuffer buffer = (cache != null) ? cache : createBackingBuffer(description, 1);
        buffer.rewind();

        // Copy the byte buffer

        byteBuffer.rewind();
        buffer.put(byteBuffer);

        return buffer;
    }

    public ByteBuffer toByteBuffer(@NonNull FloatBuffer floatBuffer, @NonNull LayerDescription description, @Nullable ByteBuffer cache) throws IllegalArgumentException {
        // Create a buffer if no reusable cache is provided

        ByteBuffer buffer = (cache != null) ? cache : createBackingBuffer(description, 1);
        buffer.rewind();

        // Copy the float buffer

        floatBuffer.rewind();
        buffer.asFloatBuffer().put(floatBuffer);

        return buffer;
    }

    public ByteBuffer toByteBuffer(@NonNull IntBuffer intBuffer, @NonNull LayerDescription description, @Nullable ByteBuffer cache) throws IllegalArgumentException {
        // Create a buffer if no reusable cache is provided

        ByteBuffer buffer = (cache != null) ? cache : createBackingBuffer(description, 1);
        buffer.rewind();

        // Copy the int buffer

        intBuffer.rewind();
        buffer.asIntBuffer().put(intBuffer);

        return buffer;
    }

    public ByteBuffer toByteBuffer(@NonNull LongBuffer longBuffer, @NonNull LayerDescription description, @Nullable ByteBuffer cache) throws IllegalArgumentException {
        // Create a buffer if no reusable cache is provided

        ByteBuffer buffer = (cache != null) ? cache : createBackingBuffer(description, 1);
        buffer.rewind();

        // Copy the long buffer

        longBuffer.rewind();
        buffer.asLongBuffer().put(longBuffer);

        return buffer;
    }

    /**
     * Processes the model output and returns an appropriate view on the output ByteBuffer given
     * the layer's data type, e.g. a [ByteBuffer], [FloatBuffer], [IntBuffer], etc.
     *
     * @param buffer      A ByteBuffer read from a TensorFlow model
     * @param description A description of the layer with instructions on how to make the conversion
     * @return The model's output ByteBuffer or a view on it
     */

    @Override
    public Object fromByteBuffer(@NonNull ByteBuffer buffer, @NonNull LayerDescription description) {

        // Acquire needed properties from layer description

        DataType dtype = ((StringLayerDescription)description).getDtype();

        // Prepare buffer for reading

        buffer.rewind();

        // Fork on type returning view on buffer

        Buffer view = buffer;

        switch (dtype) {
            case UInt8:
                view = buffer;
                break;
            case Int32:
                view = buffer.asIntBuffer();
                break;
            case Int64:
                view = buffer.asLongBuffer();
                break;
            case Float32:
                view = buffer.asFloatBuffer();
                break;
            case Unknown:;
                // Unknown assumes float
                view = buffer.asFloatBuffer();
                break;
        }

        return view;
    }

    private static IllegalArgumentException BadInputException() {
        return new IllegalArgumentException("Expected float[], byte[], int[], long[] or equivalent ByteBuffer as input to the converter");
    }
}
