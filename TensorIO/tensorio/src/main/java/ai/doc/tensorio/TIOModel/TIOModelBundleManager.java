/*
 * TIOModelBundleManager.java
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

package ai.doc.tensorio.TIOModel;

import android.content.Context;
import android.util.Log;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The `TIOModelBundleManager` manages model bundles in a provided directory. Use the returned
 * `TIOModelBundle` classes to instantiante `TIOModel` objects.
 */

public class TIOModelBundleManager {

    private Map<String, TIOModelBundle> modelBundles;

    /**
     * Loads the available models at the specified path, e.g. folders that end in .tfbundle
     * and assigns them to the models property. Models will be sorted by name by default.
     *
     * @param path Directory where model bundles are located, may be in the application bundle,
     *             application documents directory, or elsewhere.
     */

    public TIOModelBundleManager(Context c, String path) throws IOException {
        modelBundles = new HashMap<>();
        String[] assets = c.getAssets().list("");
        for(String s: assets){
            if (s.endsWith(".tfbundle")){
                try {
                    TIOModelBundle bundle = new TIOModelBundle(c, s);
                    modelBundles.put(bundle.getIdentifier(), bundle);
                } catch (TIOModelBundleException e) {
                    Log.i("TIOModelBundleManager", "Invalid bundle: "+s);
                    e.printStackTrace();
                }
            }
        }
    }

    /**
     * Returns the models that match the provided ids.
     *
     * @param modelIds Array of model ids in `String` format
     * @return List of `TIOModelBundle` matching the model ids
     */

    public List<TIOModelBundle> bundlesWithIds(String[] modelIds) {
        List<TIOModelBundle> bundles = new ArrayList<>(modelIds.length);
        for (String s: modelIds){
            bundles.add(modelBundles.get(s));
        }
        return bundles;
    }

    /**
     * Returns the single model that matches the provided id.
     * <p>
     * You must call `loadModelsAtPath:error:` before calling this method.
     *
     * @param modelId The single model id whose bundle you would like.
     * @return The `TIOModelBundle` matching the model id.
     */

    public TIOModelBundle bundleWithId(String modelId) {
        return modelBundles.get(modelId);
    }

    /**
     *
     * @return a Set of String IDs of all bundles known to the manager.
     */

    public Set<String> getBundleIds(){
        return modelBundles.keySet();
    }
}

