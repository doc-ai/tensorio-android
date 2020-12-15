/*
 * VectorConverter.java
 * TensorIO
 *
 * Created by Sam Leroux on 12/15/2020
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


package ai.doc.tensorio.pytorch.data;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import org.pytorch.Tensor;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;

import ai.doc.tensorio.core.data.Dequantizer;
import ai.doc.tensorio.core.data.Quantizer;
import ai.doc.tensorio.core.layerinterface.DataType;
import ai.doc.tensorio.core.layerinterface.LayerDescription;
import ai.doc.tensorio.core.layerinterface.VectorLayerDescription;

public class VectorConverter implements Converter {

    @Override
    public ByteBuffer createBackingBuffer(LayerDescription description) {
        boolean quantized = ((VectorLayerDescription)description).isQuantized();
        int length = ((VectorLayerDescription)description).getLength();
        DataType dtype = description.getDtype();

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
        }

        if (quantized) {
            // Override for a quantized layer
            bufferLength = length;
        }

        // Create buffer

        ByteBuffer buffer = ByteBuffer.allocateDirect(bufferLength);
        buffer.order(ByteOrder.nativeOrder());

        return buffer;


    }

    @Override
    public Tensor toTensor(@NonNull Object o, @NonNull LayerDescription description, @Nullable ByteBuffer cache) throws IllegalArgumentException {
        Tensor t;
        ByteBuffer buffer;

        int[] shape = ((VectorLayerDescription)description).getShape();
        long[] longShape = Arrays.stream(shape).asLongStream().toArray();


        if (o instanceof byte[]) {
            buffer =  toByteBuffer((byte[])o, description, cache);
            t = Tensor.fromBlob(buffer, longShape);
        }
        else if (o instanceof int[]){
            buffer = toByteBuffer((int[])o, description, cache);
            IntBuffer intBuffer = buffer.asIntBuffer();
            t = Tensor.fromBlob(buffer.asIntBuffer(), longShape);
        }
        else if (o instanceof float[]) {
            buffer =  toByteBuffer((float[])o, description, cache);
            t = Tensor.fromBlob(buffer.asFloatBuffer(), longShape);
        }
        else if (o instanceof long[]) {
            buffer =  toByteBuffer((long[])o, description, cache);
            t = Tensor.fromBlob(buffer.asLongBuffer(), longShape);
        }
        else {
            throw BadInputException();
        }


        return t;
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

    public ByteBuffer toByteBuffer(@NonNull byte[] bytes, @NonNull LayerDescription description, @Nullable ByteBuffer cache) throws IllegalArgumentException {
        // Create a buffer if no reusable cache is provided

        ByteBuffer buffer = (cache != null) ? cache : createBackingBuffer(description);
        buffer.rewind();

        // Acquire needed properties from layer description

        VectorLayerDescription vectorLayerDescription = (VectorLayerDescription) description;
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
     * Writes an array of ints to a ByteBuffer and returns it, using the cache if one is provided.
     *
     * @param ints array of ints to write to ByteBuffer
     * @param description A description of the layer with instructions on how to make the conversion
     * @param cache A pre-existing byte buffer to use, which will be returned if not null. If a cache
     *              is provided it will be rewound before being used.
     * @return ByteBuffer ready for use with a TFLite model
     */

    public ByteBuffer toByteBuffer(@NonNull int[] ints, @NonNull LayerDescription description, @Nullable ByteBuffer cache) throws IllegalArgumentException {
        // Create a buffer if no reusable cache is provided

        ByteBuffer buffer = (cache != null) ? cache : createBackingBuffer(description);
        buffer.rewind();

        // Acquire needed properties from layer description

        VectorLayerDescription vectorLayerDescription = (VectorLayerDescription) description;
        int length = vectorLayerDescription.getLength();

        // Validate input

        if (ints.length != length) {
            throw BadLengthException(ints.length, length);
        }

        // Write the bytes

        IntBuffer intBuffer = buffer.asIntBuffer();
        intBuffer.put(ints);

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

    public ByteBuffer toByteBuffer(@NonNull float[] floats, @NonNull LayerDescription description, @Nullable ByteBuffer cache) throws IllegalArgumentException {
        // Create a buffer if no reusable cache is provided

        ByteBuffer buffer = (cache != null) ? cache : createBackingBuffer(description);
        buffer.rewind();

        // Acquire needed properties from layer description

        VectorLayerDescription vectorLayerDescription = (VectorLayerDescription) description;
        Quantizer quantizer = vectorLayerDescription.getQuantizer();
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

    /**
     * Writes an array of int64s (long) to a ByteBuffer and returns it, using the cache if one is provided.
     *
     * @param longs array of int64s to write to ByteBuffer
     * @param description A description of the layer with instructions on how to make the conversion
     * @param cache A pre-existing byte buffer to use, which will be returned if not null. If a cache
     *              is provided it will be rewound before being used.
     * @return ByteBuffer ready for use with a TensorFlow model
     */

    public ByteBuffer toByteBuffer(@NonNull long[] longs, @NonNull LayerDescription description, @Nullable ByteBuffer cache) throws IllegalArgumentException {
        // Create a buffer if no reusable cache is provided

        ByteBuffer buffer = (cache != null) ? cache : createBackingBuffer(description);
        buffer.rewind();

        // Acquire needed properties from layer description

        VectorLayerDescription vectorLayerDescription = (VectorLayerDescription) description;
        int length = vectorLayerDescription.getLength();

        // Validate input

        if (longs.length != length) {
            throw BadLengthException(longs.length, length);
        }

        // Write the longs

        LongBuffer i = buffer.asLongBuffer();
        i.put(longs);

        return buffer;
    }


    @Override
    public Object fromTensor(@NonNull Tensor t, @NonNull LayerDescription description) {
        // Acquire needed properties from layer description

        VectorLayerDescription vectorLayerDescription = (VectorLayerDescription) description;
        Dequantizer dequantizer = vectorLayerDescription.getDequantizer();
        boolean quantized = vectorLayerDescription.isQuantized();
        int length = vectorLayerDescription.getLength();

        DataType dtype = description.getDtype();


        if (quantized && dequantizer == null) {
            // DataType.UInt8
            return t.getDataAsByteArray();
        } else if (quantized) {
            float[] result = new float[length];
            byte[] buffer = t.getDataAsByteArray();
            for (int i = 0; i < length; i++) {
                result[i] = dequantizer.dequantize((int) ( buffer[i] & 0xFF ) );
            }
            return result;
        } else if (dtype == DataType.Int32) {
            return t.getDataAsIntArray();
        } else if (dtype == DataType.Int64) {
            return t.getDataAsLongArray();
        } else {
            return t.getDataAsFloatArray();
        }

    }

    private static IllegalArgumentException BadInputException() {
        return new IllegalArgumentException("Expected float[] or byte[] as input to the converter");
    }

    private static IllegalArgumentException BadLengthException(int given, int expected) {
        return new IllegalArgumentException("Provided input is of different size than the size expected by the model, expected " + expected + " input has length " + given);
    }

    private static IllegalArgumentException MissingQuantizeException() {
        return new IllegalArgumentException("Float[] given as input to quantized model without quantizer, expected byte[] or quantizer != null");
    }


}
