/*
 * MainActivity.kt
 * TensorIO
 *
 * Created by Philip Dow on 7/8/2020
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

package com.docai.kotlinexample

import ai.doc.tensorio.TIOModel.TIOModelBundle
import ai.doc.tensorio.TIOModel.TIOModelBundleException
import ai.doc.tensorio.TIOModel.TIOModelException
import ai.doc.tensorio.TIOUtilities.TIOClassificationHelper
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import androidx.appcompat.app.AppCompatActivity
import android.util.Log
import android.widget.ImageView
import android.widget.TextView
import java.io.IOException

private const val TAG = "MainActivity"

class MainActivity : AppCompatActivity() {

    private val main by lazy { Handler(Looper.getMainLooper()) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        try {
            // Load the Model

            val bundle = TIOModelBundle(applicationContext, "mobilenet_v2_1.4_224.tiobundle")
            val model = bundle.newModel()

            // Load the Image

            val stream = assets.open("elephant.jpg")
            val bitmap = BitmapFactory.decodeStream(stream)
            val scaled = Bitmap.createScaledBitmap(bitmap, 224, 224, false)

            val imageView = findViewById<ImageView>(R.id.imageView)
            imageView.setImageBitmap(bitmap)

            stream.close()

            // Create a Background Thread

            val mHandlerThread = HandlerThread("HandlerThread")
            mHandlerThread.start()
            val mHandler = Handler(mHandlerThread.looper)

            // Execute the Model

            mHandler.post {
                try {
                    val output = model.runOn(scaled)
                    val classification = output.get("classification") as MutableMap<String, Float>
                    val top5 = TIOClassificationHelper.topN(classification, 5, 0.1f)

                    for (entry in top5) {
                        Log.i(TAG, entry.key + ":" + entry.value)
                    }

                    main.post {
                        val textView = findViewById<TextView>(R.id.textView)
                        textView.text = formattedResults(top5)
                    }

                } catch (e: TIOModelException) {
                    e.printStackTrace()
                }
            }
        } catch (e: IOException) {
            e.printStackTrace()
        } catch (e: TIOModelBundleException) {
            e.printStackTrace()
        }
    }

    private fun formattedResults(results: List<Map.Entry<String, Float>>): String? {
        val b = StringBuilder()

        for ((key, value) in results) {
            b.append(key)
            b.append(": ")
            b.append(value)
            b.append("\n")
        }

        b.setLength(b.length - 1)

        return b.toString()
    }
}
