import React, { useRef, useState, ChangeEvent } from 'react';
import * as tf from '@tensorflow/tfjs';

const ImageProcessor: React.FC = () => {
    const imageRef = useRef<HTMLInputElement>(null);
    const modelRef = useRef<HTMLInputElement>(null);
    const weightsRef = useRef<HTMLInputElement>(null);
    const [modelFile, setModelFile] = useState<File | null>(null);
    const [weightsFile, setWeightsFile] = useState<File | null>(null);
    const [prediction, setPrediction] = useState<string | null>(null);
    const [imageUrl, setImageUrl] = useState<string | null>(null);

    const handleImageUpload = (e: ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files ? e.target.files[0] : null;
        if (file) {
            const reader = new FileReader();
            reader.onload = () => {
                setImageUrl(reader.result as string);
            };
            reader.readAsDataURL(file);
        }
    };


    const handleModelUpload = async (e: ChangeEvent<HTMLInputElement>) => {
        // Load the model using TensorFlow.js
        const uploadedModel = e.target.files ? e.target.files[0] : null;
        if (uploadedModel) {
            setModelFile(uploadedModel);
        }
    };

    const handleWeightsUpload = async (e: ChangeEvent<HTMLInputElement>) => {
        const uploadedWeights = e.target.files ? e.target.files[0] : null;
        if (uploadedWeights) {
            setWeightsFile(uploadedWeights);
        }
    }


    const handlePrediction = async () => {
        if (modelFile && weightsFile && imageUrl) {
            const image = new Image();
            image.src = imageUrl;
            await new Promise((resolve) => {
                image.onload = resolve;
            });

            let tensor = tf.browser.fromPixels(image).toFloat();

            // Redimensionnement de l'image à 224x224
            tensor = tf.image.resizeBilinear(tensor, [224, 224]);

            // Ajout d'une dimension pour le lot
            tensor = tensor.expandDims(0);

            const model = await tf.loadGraphModel(tf.io.browserFiles([modelFile, weightsFile]));
            const prediction = await model.predict(tensor);
            let predictionData;

            if (prediction instanceof tf.Tensor) {
                predictionData = await prediction.data();
            } else {
                return;
            }

            // Utilisation des données de prédictions
            console.log(predictionData);

            // Si la sortie est une classification, trouver l'index de la classe avec la valeur la plus élevée :
            const predictedClassIdx = predictionData.indexOf(Math.max(...predictionData));
            console.log(`Predicted class index: ${predictedClassIdx}`);

            setPrediction(predictedClassIdx.toString());
        }
    };



    return (
        <div>
            <p>Image : </p>
            <input type="file" accept="image/*" ref={imageRef} onChange={handleImageUpload} />
            <p>Model : </p>
            <input type="file" accept=".json" ref={modelRef} onChange={handleModelUpload} /><br/>
            <p>Weights : </p>
            <input type="file" accept=".bin" ref={weightsRef} onChange={handleWeightsUpload} /><br/>
            <button onClick={handlePrediction}>Predict</button><br/>
            {prediction && <div>Result: {prediction}</div>}
        </div>
    );
};

export default ImageProcessor;
