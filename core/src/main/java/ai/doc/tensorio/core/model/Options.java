/*
 * ModelOptions.java
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

import androidx.annotation.NonNull;

/**
 * Encapsulates additional options that a model would like to communicate to it consumers.
 */

public class Options {

    /**
     * Preferred device position.
     * <p>
     * If the device position is unspecified at initialization, `0` will be used,
     * which will then typically default to the back facing camera.
     */

    private String devicePosition;

    /**
     * Converts a string representation of a capture device position, e.g. a camera position,
     * to a Camera ID
     *
     * @param descriptor A string representation of a device position. 'front' and 'back' are
     *                   the only values currently supported.
     */

    public Options(@NonNull String descriptor) {
        if (descriptor.equals("front")) {
            this.devicePosition = "1";
        } else {
            this.devicePosition = "0";
        }
    }

    public String getDevicePosition() {
        return devicePosition;
    }
}
