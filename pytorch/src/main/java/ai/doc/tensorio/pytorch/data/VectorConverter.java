package ai.doc.tensorio.pytorch.data;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import org.pytorch.Tensor;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;

import ai.doc.tensorio.core.data.Dequantizer;
import ai.doc.tensorio.core.data.Quantizer;
import ai.doc.tensorio.core.layerinterface.LayerDescription;
import ai.doc.tensorio.core.layerinterface.VectorLayerDescription;

public class VectorConverter implements Converter {

    @Override
    public ByteBuffer createBackingBuffer(LayerDescription description) {
        ByteBuffer buffer;

        boolean quantized = ((VectorLayerDescription)description).isQuantized();
        int length = ((VectorLayerDescription)description).getLength();

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
    public Tensor toTensor(@NonNull Object o, @NonNull LayerDescription description, @Nullable ByteBuffer cache) throws IllegalArgumentException {
        Tensor t;
        ByteBuffer buffer;

        int[] shape = ((VectorLayerDescription)description).getShape();
        long[] longShape = Arrays.stream(shape).asLongStream().toArray();


        if (o instanceof byte[]) {
            buffer =  toByteBuffer((byte[])o, description, cache);
            t = Tensor.fromBlob(buffer, longShape);
        } else if (o instanceof float[]) {
            buffer =  toByteBuffer((float[])o, description, cache);
            t = Tensor.fromBlob(buffer.asFloatBuffer(), longShape);
        } else {
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


    @Override
    public Object fromTensor(@NonNull Tensor t, @NonNull LayerDescription description) {
        // Acquire needed properties from layer description

        VectorLayerDescription vectorLayerDescription = (VectorLayerDescription) description;
        Dequantizer dequantizer = vectorLayerDescription.getDequantizer();
        boolean quantized = vectorLayerDescription.isQuantized();
        int length = vectorLayerDescription.getLength();

        if (quantized && dequantizer == null){
            return t.getDataAsByteArray();
        }
        else if (quantized){
            float[] result = new float[length];
            byte[] buffer = t.getDataAsByteArray();
            for (int i = 0; i < length; i++) {
                result[i] = dequantizer.dequantize((int) ( buffer[i] & 0xFF ) );
            }
            return result;
        }
        else{
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
