package ai.doc.tensorio.pytorch.data;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import org.pytorch.Tensor;

import java.nio.ByteBuffer;

import ai.doc.tensorio.core.layerinterface.LayerDescription;

public class StringConverter implements Converter {
    @Override
    public Tensor toTensor(@NonNull Object o, @NonNull LayerDescription description, @Nullable ByteBuffer cache) throws IllegalArgumentException {
        return null;
    }

    @Override
    public Object fromTensor(@NonNull Tensor t, @NonNull LayerDescription description) {
        return null;
    }

    @Override
    public ByteBuffer createBackingBuffer(LayerDescription stringLayer) {
        return null;
    }
}
