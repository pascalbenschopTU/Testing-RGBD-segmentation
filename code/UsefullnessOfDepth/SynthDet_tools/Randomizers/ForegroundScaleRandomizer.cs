using System;
using SynthDet.RandomizerTags;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;

namespace SynthDet.Randomizers
{
    [Serializable]
    [AddRandomizerMenu("SynthDet/Foreground Scale Randomizer for Objects With Normalized Bounds")]
    public class ForegroundScaleRandomizer : Randomizer
    {
        [Tooltip("The scale factor to apply to the object's scale. The object's scale will be multiplied by a random value between the min and max values specified here.")]
        public FloatParameter scale = new FloatParameter { value = new UniformSampler(0.5f, 0.5f) };

        [Tooltip("If true, the scale factor will be randomly applied to each individual axis. If false, the scale factor will be sampled once and applied to all axes.")]
        public bool applyToAllAxes = false; 

        protected override void OnIterationStart()
        {
            var seed = SamplerState.NextRandomState();
            UnityEngine.Random.InitState((int)seed);
            
            var tags = tagManager.Query<ForegroundScaleRandomizerTag>();
            
            foreach (var tag in tags)
            {
                if (applyToAllAxes)
                {
                    Vector3 scaleVector = new Vector3(scale.Sample(), scale.Sample(), scale.Sample());
                    Vector3 currentScale = tag.transform.localScale;
                    tag.transform.localScale = new Vector3(currentScale.x * scaleVector.x, currentScale.y * scaleVector.y, currentScale.z * scaleVector.z);
                }
                else
                {
                    tag.transform.localScale *= scale.Sample();
                }
            }
        }
    }
}