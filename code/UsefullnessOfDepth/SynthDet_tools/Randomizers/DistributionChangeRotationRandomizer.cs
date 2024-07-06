using System;
using SynthDet.RandomizerTags;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;

namespace SynthDet.Randomizers
{
    /// <summary>
    /// Randomizes the rotation of objects tagged with a RotationRandomizerTag
    /// </summary>
    [Serializable]
    [AddRandomizerMenu("SynthDet/Distribution Change Rotation Randomizer")]
    public class DistributionChangeRotationRandomizer : Randomizer
    {
        [Tooltip("The rotation values to sample from for training.")]
        public Vector3Parameter trainRotation = new Vector3Parameter
        {
            x = new UniformSampler(0, 360),
            y = new UniformSampler(0, 360),
            z = new UniformSampler(0, 360)
        };

        [Tooltip("If true, the rotations in the first half of the iterations will be cubic rotations.")]
        public bool cubicRotationsInTrainSubset = false;

        [Tooltip("The minimum and maximum rotation values for the test subset. The rotation values will be linearly interpolated between these values over the course of the test subset iterations.")]
        public int test_rotation_min_x = 0;
        public int test_rotation_max_x = 360;
        public int test_rotation_min_y = 0;
        public int test_rotation_max_y = 360;
        public int test_rotation_min_z = 0;
        public int test_rotation_max_z = 360;

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
            InitializeRandomSeed();
            
            var rotationSample = CalculateRotationSample();
            ApplyRotationToTags(rotationSample);
        }

        private void InitializeRandomSeed()
        {
            var seed = SamplerState.NextRandomState();
            UnityEngine.Random.InitState((int)seed);
        }

        private Quaternion CalculateRotationSample()
        {
            if (IsInFirstHalfOfIterations())
            {
                return cubicRotationsInTrainSubset ? GetCubicRotation() : Quaternion.Euler(trainRotation.Sample());
            }
            else
            {
                return GetLinearInterpolatedRotation();
            }
        }

        private bool IsInFirstHalfOfIterations()
        {
            return (int)scenario.currentIteration < num_train_images;
        }

        private Quaternion GetCubicRotation()
        {
            var x = UnityEngine.Random.Range(0, 4) * 90; // Consider using a named constant for 90
            var y = UnityEngine.Random.Range(0, 4) * 90;
            var z = UnityEngine.Random.Range(0, 4) * 90;
            return Quaternion.Euler(x, y, z);
        }

        private Quaternion GetLinearInterpolatedRotation()
        {
            var percent = (float)(scenario.currentIteration - num_train_images) / (float)num_train_images;
            var x = Mathf.Lerp(test_rotation_min_x, test_rotation_max_x, percent);
            var y = Mathf.Lerp(test_rotation_min_y, test_rotation_max_y, percent);
            var z = Mathf.Lerp(test_rotation_min_z, test_rotation_max_z, percent);
            return Quaternion.Euler(x, y, z);
        }

        private void ApplyRotationToTags(Quaternion rotationSample)
        {
            var tags = tagManager.Query<UnifiedRotationRandomizerTag>();
            foreach (var tag in tags)
            {
                tag.transform.rotation = rotationSample;
            }
        }
    }
}

