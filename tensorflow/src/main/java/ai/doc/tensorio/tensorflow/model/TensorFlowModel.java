package ai.doc.tensorio.tensorflow.model;

import android.graphics.Bitmap;

import java.util.Map;

import ai.doc.tensorio.core.model.Model;
import ai.doc.tensorio.core.modelbundle.ModelBundle;
import androidx.annotation.NonNull;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class TensorFlowModel extends Model {

    public TensorFlowModel(@NonNull ModelBundle bundle) {
        super(bundle);
    }

    @Override
    public Map<String, Object> runOn(float[] input) throws ModelException {
        return null;
    }

    @Override
    public Map<String, Object> runOn(byte[] input) throws ModelException {
        return null;
    }

    @Override
    public Map<String, Object> runOn(@NonNull Bitmap input) throws ModelException {
        return null;
    }

    @Override
    public Map<String, Object> runOn(@NonNull Map<String, Object> input) throws ModelException, IllegalArgumentException {
        return null;
    }
}
