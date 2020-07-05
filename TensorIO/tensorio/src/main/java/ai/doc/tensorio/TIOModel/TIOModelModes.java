package ai.doc.tensorio.TIOModel;

import android.support.annotation.Nullable;

import org.json.JSONArray;
import org.json.JSONException;

import java.util.EnumSet;

public class TIOModelModes {

    private enum TIOModelMode {
        Predict,
        Train,
        Eval
    }

    /**
     * Parses a JSON array of strings into an EnumSet of TIOModelModes.
     * @param array JSON array of strings. May be null for backwards compatibility, in which case
     *              the mode will be interpreted as predict.
     * @return EnumSet of TIOModelModels
     * @throws JSONException
     */

    static private EnumSet<TIOModelMode> parseModelModes(@Nullable JSONArray array) throws JSONException {
        if (array == null || array.length() == 0) {
            return EnumSet.of(TIOModelMode.Predict);
        }

        EnumSet<TIOModelMode> set = EnumSet.noneOf(TIOModelMode.class);

        for (int i=0; i < array.length(); i++) {
            String mode = array.getString(i);

            if (mode.equals("predict")) {
                set.add(TIOModelMode.Predict);
            }
            if (mode.equals("train")) {
                set.add(TIOModelMode.Train);
            }
            if (mode.equals("eval")) {
                set.add(TIOModelMode.Eval);
            }
        }

        return set;
    }

    final private EnumSet<TIOModelMode> modes;

    /**
     * Designated initializer
     * @param array a JSONArray of Strings describing the model's supported modes
     * @throws JSONException
     */

    public TIOModelModes(JSONArray array) throws JSONException {
        this.modes = TIOModelModes.parseModelModes(array);
    }

    /**
     * Backwards compatible initializer when a model.json file describes no modes. Initializes
     * modes with support for predict.
     * @throws JSONException
     */

    public TIOModelModes() throws JSONException{
        this(null);
    }

    public boolean predicts() {
        return this.modes.contains(TIOModelMode.Predict);
    }

    public boolean trains() {
        return this.modes.contains(TIOModelMode.Train);
    }

    public boolean evals() {
        return this.modes.contains(TIOModelMode.Eval);
    }

}
