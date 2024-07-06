using System;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers.Tags;
using UnityEngine.Perception.Randomization.Samplers;
using UnityEngine.Scripting.APIUpdating;
using ObjectTypeARandomizerTag = SynthDet.RandomizerTags.ObjectTypeARandomizerTag;
using ObjectTypeBRandomizerTag = SynthDet.RandomizerTags.ObjectTypeBRandomizerTag;

namespace UnityEngine.Perception.Randomization.Randomizers
{
    /// <summary>
    /// Randomizes the material texture of objects tagged with a TextureRandomizerTag
    /// </summary>
    [Serializable]
    [AddRandomizerMenu("Perception/Distribution Change Texture Randomizer")]
    public class DistributionChangeTextureRandomizer : Randomizer
    {
        static readonly int k_BaseMap = Shader.PropertyToID("_BaseMap");
#if HDRP_PRESENT
        const string k_TutorialHueShaderName = "Shader Graphs/HueShiftOpaque";
        static readonly int k_BaseColorMap = Shader.PropertyToID("_BaseColorMap");
#endif
        public bool useBackgroundTextures = true;
        /// <summary>
        /// The list of textures to sample and apply to target objects
        /// </summary>
        [Tooltip("The list of textures to sample and apply to target objects.")]
        public CategoricalParameter<Texture2D> trainTextures;

        [Tooltip("The list of textures to sample and apply to target objects.")]
        public CategoricalParameter<Texture2D> testTextures;


        private int iteration_count = 1;

        protected override void OnScenarioStart() 
        {
            var constants = scenario.GetType().GetField("constants").GetValue(scenario);
            iteration_count = (int)constants.GetType().GetField("iterationCount").GetValue(constants);
        }

        /// <summary>
        /// Randomizes the material texture of tagged objects at the start of each scenario iteration
        /// </summary>
        protected override void OnIterationStart()
        {
            var seed = SamplerState.NextRandomState();
            UnityEngine.Random.InitState((int)seed);

            var tags = tagManager.Query<TextureRandomizerTag>();
            foreach (var tag in tags)
            {
                
                var renderer = tag.GetComponent<Renderer>();
#if HDRP_PRESENT
                // Choose the appropriate shader texture property ID depending on whether the current material is
                // using the default HDRP/lit shader or the Perception tutorial's HueShiftOpaque shader
                var material = renderer.material;
                var propertyId = material.shader.name == k_TutorialHueShaderName ? k_BaseMap : k_BaseColorMap;
                // Texture for all objects, regardless of class
                var texture = (int)scenario.currentIteration < (iteration_count / 2) ? trainTextures.Sample() : testTextures.Sample();
                if (!useBackgroundTextures)
                {
                    texture = null;
                }

                if (texture == null)
                    continue;
                    
                material.SetTexture(propertyId, texture);
#endif
            }
        }
    }
}