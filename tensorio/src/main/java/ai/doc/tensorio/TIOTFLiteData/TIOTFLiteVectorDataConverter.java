/*
 * TIOTFLiteVectorDataConverter.java
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

package ai.doc.tensorio.TIOTFLiteData;

import androidx.annotation.Nullable;
import androidx.annotation.NonNull;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import ai.doc.tensorio.TIOData.TIODataConverter;
import ai.doc.tensorio.TIOData.TIODataDequantizer;
import ai.doc.tensorio.TIOData.TIODataQuantizer;
import ai.doc.tensorio.TIOLayerInterface.TIOLayerDescription;
import ai.doc.tensorio.TIOLayerInterface.TIOVectorLayerDescription;

public class TIOTFLiteVectorDataConverter implements TIODataConverter, TIOTFLiteDataConverter {

    @Override
    public ByteBuffer createBackingBuffer(@NonNull TIOLayerDescription description) {
        ByteBuffer buffer;

        boolean quantized = ((TIOVectorLayerDescription)description).isQuantized();
        int length = ((TIOVectorLayerDescription)description).getLength();

        if (quantized) {
            // Layer expects bytes
            buffer = ByteBuffer.allocateDirect(length);
        } else {
            // Layer expects floats
            buffer = ByteBuffer.allocateDirect(length*4);
        }

        buffer.order(ByteOrder.nativeOrder());

        return buffer;
    }

    @Override
    public ByteBuffer toByteBuffer(@NonNull Object o, @NonNull TIOLayerDescription description, @Nullable ByteBuffer cache) throws IllegalArgumentException {
        if (o instanceof byte[]) {
            return toByteBuffer((byte[])o, description, cache);
        } else if (o instanceof float[]) {
            return toByteBuffer((float[])o, description, cache);
        } else {
            throw BadInputException();
        }
    }

    /**
     * Writes an array of bytes to a ByteBuffer and returns it, using the cache if one is provided.
     *
     * @param bytes array of bytes to write to ByteBuffer
     * @param description A description of the layer with instructions on how to make the conversion
     * @param cache A pre-existing byte buffer to use, which will be returned if not null. If a cache
     *              is provided it will be rewound before being used.
     * @return ByteBuffer ready for use with a TFLite model
     */

    public ByteBuffer toByteBuffer(@NonNull byte[] bytes, @NonNull TIOLayerDescription description, @Nullable ByteBuffer cache) throws IllegalArgumentException {
        // Create a buffer if no reusable cache is provided

        ByteBuffer buffer = (cache != null) ? cache : createBackingBuffer(description);
        buffer.rewind();

        // Acquire needed properties from layer description

        TIOVectorLayerDescription vectorLayerDescription = (TIOVectorLayerDescription) description;
        int length = vectorLayerDescription.getLength();

        // Validate input

        if (bytes.length != length) {
            throw BadLengthException(bytes.length, length);
        }

        // Write the bytes

        buffer.put(bytes);

        return buffer;
    }

    /**
     * Writes an array of floats to a ByteBuffer and returns it, quantizing the values if necessary,
     * and using the cache if one is provided.
     *
     * @param floats array of floats to write to ByteBuffer
     * @param description A description of the layer with instructions on how to make the conversion
     * @param cache A pre-existing byte buffer to use, which will be returned if not null. If a cache
     *              is provided it will be rewound before being used.
     * @return ByteBuffer ready for use with a TFLite model
     */

    public ByteBuffer toByteBuffer(@NonNull float[] floats, @NonNull TIOLayerDescription description, @Nullable ByteBuffer cache) throws IllegalArgumentException {
        // Create a buffer if no reusable cache is provided

        ByteBuffer buffer = (cache != null) ? cache : createBackingBuffer(description);
        buffer.rewind();

        // Acquire needed properties from layer description

        TIOVectorLayerDescription vectorLayerDescription = (TIOVectorLayerDescription) description;
        TIODataQuantizer quantizer = vectorLayerDescription.getQuantizer();
        boolean quantized = vectorLayerDescription.isQuantized();
        int length = vectorLayerDescription.getLength();

        // Validate input

        if (floats.length != length) {
            throw BadLengthException(floats.length, length);
        }

        // Fork on quantized and write bytes

        if (quantized && quantizer == null) {
            throw MissingQuantizeException();
        } else if (quantized) {
            for (float v: floats) {
                buffer.put((byte)quantizer.quantize(v));
            }
        } else {
            FloatBuffer f = buffer.asFloatBuffer();
            f.put(floats);
        }

        return buffer;
    }

    // Note that bytes are signed in java so when we read outputs from a ByteBuffer as bytes we
    // might get negative values. So we first have to unsign the byte with & 0xFF and then cast to int.

    @Override
    public Object fromByteBuffer(@NonNull ByteBuffer buffer, @NonNull TIOLayerDescription description) {

        // Acquire needed properties from layer description

        TIOVectorLayerDescription vectorLayerDescription = (TIOVectorLayerDescription) description;
        TIODataDequantizer dequantizer = vectorLayerDescription.getDequantizer();
        boolean quantized = vectorLayerDescription.isQuantized();
        int length = vectorLayerDescription.getLength();

        // Prepare buffer for reading

        buffer.rewind();

        // Fork on quantized

        if (quantized && dequantizer == null) {
            return buffer.array();
        } else if (quantized) {
            float[] result = new float[length];
            for (int i = 0; i < length; i++) {
                result[i] = dequantizer.dequantize((int) ( buffer.get() & 0xFF ) );
            }
            return result;
        } else {
            float[] result = new float[length];
            buffer.asFloatBuffer().get(result);
            return result;
        }
    }

    //region Exceptions

    private static IllegalArgumentException BadInputException() {
        return new IllegalArgumentException("Expected float[] or byte[] as input to the converter");
    }

    private static IllegalArgumentException BadLengthException(int given, int expected) {
        return new IllegalArgumentException("Provided input is of different size than the size expected by the model, expected " + expected + " input has length " + given);
    }

    private static IllegalArgumentException MissingQuantizeException() {
        return new IllegalArgumentException("Float[] given as input to quantized model without quantizer, expected byte[] or quantizer != null");
    }

    //endRegion
}
