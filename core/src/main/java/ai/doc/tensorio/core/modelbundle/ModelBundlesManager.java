/*
 * ModelBundleManager.java
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

package ai.doc.tensorio.core.modelbundle;

import android.content.Context;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

/**
 * The `ModelBundleManager` manages model bundles in a provided directory. Use the returned
 * `ModelBundle` classes to instantiate `Model` objects.
 *
 * This is an abstract class. Use one of the static methods @see managerWithAssets or
 * @see managerWithFiles to get a concrete instance from an assets or File directory.
 */

public abstract class ModelBundlesManager {

    /**
     * Creates and returns a new AssetModelBundlesManager for the bundles in the assets directory at path
     *
     * @param c The application or activity context
     * @param path The path to the directory containing the model bundles in the assets, including subdirectories
     * @return A @see AssetModelBundlesManager
     * @throws IOException On any error reading the directory contents
     */

    public static ModelBundlesManager managerWithAssets(@NonNull Context c, @NonNull String path) throws IOException {
        return new AssetModelBundlesManager(c, path);
    }

    /**
     * Creates and returns a new FileModelBundlesManager for the bundles in the File directory
     *
     * @param file The File directory containing the model bundles
     * @return A @see FileModelBundlesManager
     * @throws IOException On any error reading the directory contents
     */

    public static ModelBundlesManager managerWithFiles(@NonNull File file) throws IOException {
        return new FileModelBundlesManager(file);
    }

    /** Map of Model Bundle identifiers to Model Bundles */

    protected Map<String, ModelBundle> modelBundles;

    /**
     * Reload the managed model bundles
     */

    public abstract void reload();

    /**
     * Returns the models that match the provided ids.
     *
     * @param modelIds Array of model ids in `String` format
     * @return List of `ModelBundle` matching the model ids
     */

    public List<ModelBundle> bundlesWithIds(@NonNull String[] modelIds) {
        List<ModelBundle> bundles = new ArrayList<>(modelIds.length);

        for (String s: modelIds){
            bundles.add(modelBundles.get(s));
        }

        return bundles;
    }

    /**
     * Returns the single model that matches the provided id.
     *
     * @param modelId The single model id whose bundle you would like.
     * @return The `ModelBundle` matching the model id.
     */

    public @Nullable
    ModelBundle bundleWithId(@NonNull String modelId) {
        return modelBundles.get(modelId);
    }

    /**
     * @return a Set of String IDs of all bundles known to the manager.
     */

    public Set<String> getBundleIds() {
        return modelBundles.keySet();
    }
}

