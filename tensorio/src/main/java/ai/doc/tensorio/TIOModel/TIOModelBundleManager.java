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

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

/**
 * The `TIOModelBundleManager` manages model bundles in a provided directory. Use the returned
 * `TIOModelBundle` classes to instantiate `TIOModel` objects.
 */

public class TIOModelBundleManager {

    /** Source is an asset from a context or a file. Barf */

    public enum Source {
        Asset,
        File
    };

    private Source source;

    /** Context and path are used when the manager targets assets in an application package */

    private Context context;
    private String path;

    /** File is used when the manager targets files at an arbitrary file location */

    private File file;

    /** Map of Model Bundle identifiers to Model Bundles */

    private Map<String, TIOModelBundle> modelBundles;

    /**
     * Loads the available models at the relative path in context, e.g. folders that end in .tiobundle or
     * the now deprecated .tfbundle, and assigns them to the models property. Models will be sorted
     * by name by default.
     *
     * @param c The `Context` containing an assets folder and models at path
     * @param path Directory in the context's assets folder where model bundles are located. Only
     *             a shallow search is performed.
     */

    public TIOModelBundleManager(@NonNull Context c, @NonNull String path) throws IOException {
        this.source = Source.Asset;
        this.context = c;
        this.path = path;

        loadAssets();
    }

    private void loadAssets() throws IOException {
        modelBundles = new HashMap<>();

        String[] assets = context.getAssets().list(path);

        if (assets == null) {
            return;
        }

        for (String s: assets) {
            if ( !(s.endsWith(TIOModelBundle.TF_BUNDLE_EXTENSION) || s.endsWith(TIOModelBundle.TIO_BUNDLE_EXTENSION)) ) {
                continue;
            }

            try {
                TIOModelBundle bundle = new TIOModelBundle(context, s);
                modelBundles.put(bundle.getIdentifier(), bundle);
            } catch (TIOModelBundleException e) {
                Log.i("TIOModelBundleManager", "Invalid bundle: " + s);
                e.printStackTrace();
            }
        }
    }

    /**
     * Loads the available models in the directory specified by the file, e.g. folders that end in
     * .tiobundle or the now deprecated .tfbundle, and assigns them to the models property. Models
     * will be sorted by name by default.
     *
     * @param file The directory containing the model bundles. Only a shallow search is performed.
     * @throws IOException
     */

    public TIOModelBundleManager(@NonNull File file) throws IOException {
        if (!file.isDirectory()) {
            throw new FileNotFoundException("Not a directory");
        }

        this.source = Source.File;
        this.file = file;

        loadFiles();
    }

    private void loadFiles() {
        modelBundles = new HashMap<>();

        FilenameFilter filter = (dir, name) -> name.endsWith(TIOModelBundle.TIO_BUNDLE_EXTENSION) || name.endsWith(TIOModelBundle.TF_BUNDLE_EXTENSION);
        File[] contents = file.listFiles(filter);

        for (File f : contents) {
            try {
                TIOModelBundle bundle = new TIOModelBundle(f);
                modelBundles.put(bundle.getIdentifier(), bundle);
            } catch (TIOModelBundleException e) {
                Log.i("TIOModelBundleManager", "Invalid bundle: " + f.getPath());
                e.printStackTrace();
            }
        }
    }

    /** Reload the managed model bundles */

    public void reload() {
        try {
            switch (source) {
                case Asset:
                    loadAssets();
                    break;
                case File:
                    loadFiles();
                    break;
            }
        } catch (IOException e) {
            // This should never happen, initialization would have already caught the exception
            Log.e("TIOModelBundleManager", "Unexpected IO Exception loading assets: " + e.getMessage());
        }

    }

    /**
     * Returns the models that match the provided ids.
     *
     * @param modelIds Array of model ids in `String` format
     * @return List of `TIOModelBundle` matching the model ids
     */

    public List<TIOModelBundle> bundlesWithIds(@NonNull String[] modelIds) {
        List<TIOModelBundle> bundles = new ArrayList<>(modelIds.length);

        for (String s: modelIds){
            bundles.add(modelBundles.get(s));
        }

        return bundles;
    }

    /**
     * Returns the single model that matches the provided id.
     *
     * @param modelId The single model id whose bundle you would like.
     * @return The `TIOModelBundle` matching the model id.
     */

    public @Nullable TIOModelBundle bundleWithId(@NonNull String modelId) {
        return modelBundles.get(modelId);
    }

    /**
     * @return a Set of String IDs of all bundles known to the manager.
     */

    public Set<String> getBundleIds() {
        return modelBundles.keySet();
    }
}

