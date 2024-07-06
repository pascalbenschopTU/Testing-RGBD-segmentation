using System;
using System.Linq;
using System.Collections.Generic;
using SynthDet.RandomizerTags;
using UnityEngine;
using UnityEngine.Perception.Randomization;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;
using UnityEngine.Perception.Randomization.Utilities;

namespace SynthDet.Randomizers
{
    [Serializable]
    [AddRandomizerMenu("SynthDet/Foreground Color Randomizer")]
    public class ForegroundColorRandomizer : Randomizer
    {
        [Tooltip("If true, a new color will be sampled for each object. If false, the same color will be applied to all objects.")]
        public bool perSample = true;

        [Tooltip("The colors to sample from for training.")]
        public CategoricalParameter<Material> trainColors;

        [Tooltip("The colors to sample from for testing.")]
        public CategoricalParameter<Material> testColors;

        private int iteration_count = 1;

        [Tooltip("The number of training images to generate. The remaining images will be test images.")]
        public int num_train_images = 200;

        protected override void OnScenarioStart() 
        {
            var constants = scenario.GetType().GetField("constants").GetValue(scenario);
            iteration_count = (int)constants.GetType().GetField("iterationCount").GetValue(constants);
        }

        protected override void OnIterationStart()
        {
            var seed = SamplerState.NextRandomState();
            UnityEngine.Random.InitState((int)seed);
            
            var tags = tagManager.Query<ForegroundScaleRandomizerTag>();
            
            var color = trainColors.Sample();
            if ((int)scenario.currentIteration >= num_train_images)
            {
                color = testColors.Sample();
            }
            

            foreach (var tag in tags)
            {
                var renderer = tag.GetComponent<Renderer>();
                if (renderer != null)
                {
                    if (perSample)
                    {
                        color = trainColors.Sample();
                        if ((int)scenario.currentIteration >= num_train_images)
                        {
                            color = testColors.Sample();
                        }
                    } 

                    renderer.material = color;
                }
            }
        }
    }
}