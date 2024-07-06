using System;
using SynthDet.RandomizerTags;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;

namespace SynthDet.Randomizers
{
    [Serializable]
    [AddRandomizerMenu("SynthDet/Foreground Occluder Scale Randomizer")]
    public class ForegroundOccluderScaleRandomizer : Randomizer
    {
        [Tooltip("The scale factor to apply to the object's scale. The object's scale will be multiplied by a random value between the min and max values specified here.")]
        public FloatParameter scale = new FloatParameter { value = new UniformSampler(0.5f, 6f) };

        protected override void OnIterationStart()
        {
            var seed = SamplerState.NextRandomState();
            UnityEngine.Random.InitState((int)seed);
            
            var tags = tagManager.Query<ForegroundOccluderScaleRandomizerTag>();
            foreach (var tag in tags)
            {
                tag.transform.localScale = Vector3.one * scale.Sample();
            }
        }
    }
}