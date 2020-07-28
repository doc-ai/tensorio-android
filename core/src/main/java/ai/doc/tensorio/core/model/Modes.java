/*
 * ModelModes.java
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

package ai.doc.tensorio.core.model;

import androidx.annotation.Nullable;

import org.json.JSONArray;
import org.json.JSONException;

import java.util.EnumSet;

public class Modes {

    private enum ModelMode {
        Predict,
        Train,
        Eval
    }

    /**
     * Parses a JSON array of strings into an EnumSet of ModelModes.
     * @param array JSON array of strings. May be null for backwards compatibility, in which case
     *              the mode will be interpreted as predict.
     * @return EnumSet of ModelModes
     * @throws JSONException
     */

    static private EnumSet<ModelMode> parseModelModes(@Nullable JSONArray array) throws JSONException {
        if (array == null || array.length() == 0) {
            return EnumSet.of(ModelMode.Predict);
        }

        EnumSet<ModelMode> set = EnumSet.noneOf(ModelMode.class);

        for (int i=0; i < array.length(); i++) {
            String mode = array.getString(i);

            if (mode.equals("predict")) {
                set.add(ModelMode.Predict);
            }
            if (mode.equals("train")) {
                set.add(ModelMode.Train);
            }
            if (mode.equals("eval")) {
                set.add(ModelMode.Eval);
            }
        }

        return set;
    }

    final private EnumSet<ModelMode> modes;

    /**
     * Designated initializer
     * @param array a JSONArray of Strings describing the model's supported modes
     * @throws JSONException
     */

    public Modes(JSONArray array) throws JSONException {
        this.modes = Modes.parseModelModes(array);
    }

    /**
     * Backwards compatible initializer when a model.json file describes no modes. Initializes
     * modes with support for predict.
     * @throws JSONException
     */

    public Modes() throws JSONException{
        this(null);
    }

    public boolean predicts() {
        return this.modes.contains(ModelMode.Predict);
    }

    public boolean trains() {
        return this.modes.contains(ModelMode.Train);
    }

    public boolean evals() {
        return this.modes.contains(ModelMode.Eval);
    }

}
