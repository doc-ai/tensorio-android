package ai.doc.tensorio.core.model;

import java.util.HashMap;
import java.util.Map;

import androidx.annotation.Nullable;

public class Backend {

    private static final String TF_LITE_BACKEND =       "tflite";
    private static final String TENSORFLOW_BACKEND =    "tensorflow";

    private static final String TF_LITE_MODEL_CLASS_NAME =      "ai.doc.tensorio.tflite.model.TFLiteModel";
    private static final String TENSORFLOW_MODEL_CLASS_NAME =   "ai.doc.tensorio.tensorflow.model.TensorFlowModel";

    private static final String[] classnames = {
            TF_LITE_MODEL_CLASS_NAME,
            TENSORFLOW_MODEL_CLASS_NAME
    };

    private static Map<String, String> backendToClassNames = null;
    private static Map<String, String> classNameToBackend = null;

    /**
     * Walks down the list of classnames defined above looking for the first available one
     * @return The available backend given the existence of a model class for that backend
     */

    @Nullable
    public static String availableBackend() {
        if (classNameToBackend == null) {
            classNameToBackend = new HashMap<>();
            classNameToBackend.put(TF_LITE_MODEL_CLASS_NAME, TF_LITE_BACKEND);
            classNameToBackend.put(TENSORFLOW_MODEL_CLASS_NAME, TENSORFLOW_BACKEND);
        }

        for (String name : classnames) {
            try {
                Class.forName(name);
                return classNameToBackend.get(name);
            } catch (Exception e) {
            }
        }

        return null;
    }

    /**
     * Returns the default model classname for a given backend
     * @param backend The backend
     * @return The model classname
     */

    public static String classNameForBackend(String backend) {
        if (backendToClassNames == null) {
            backendToClassNames = new HashMap<>();
            backendToClassNames.put(TF_LITE_BACKEND, TF_LITE_MODEL_CLASS_NAME);
            backendToClassNames.put(TENSORFLOW_BACKEND, TENSORFLOW_MODEL_CLASS_NAME);
        }

        return backendToClassNames.get(backend);
    }
}
