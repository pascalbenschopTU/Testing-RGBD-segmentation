using System;
using System.Linq;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;
using UnityEngine.Perception.Randomization.Utilities;

namespace SynthDet.Randomizers
{
    /// <summary>
    /// Creates a 2D layer of of evenly spaced GameObjects from a given list of prefabs
    /// </summary>
    [Serializable]
    [AddRandomizerMenu("SynthDet/Foreground Occluder Placement Randomizer")]
    public class ForegroundOccluderPlacementRandomizer : Randomizer
    {
     
        [Tooltip("The Z offset component applied to all generated foreground objects.")]
        public FloatParameter depth = new FloatParameter { value = new UniformSampler(4.0f, 7.0f) };

        [Tooltip("The scale factor to apply to the object's scale. The object's scale will be multiplied by a random value between the min and max values specified here.")]
		public FloatParameter scale = new FloatParameter { value = new UniformSampler(4f, 6f) };

        [Tooltip("The minimum distance between objects in the 2D plane.")]
        public FloatParameter occluderSeparationDistance = new FloatParameter { value = new UniformSampler(1f, 1f) };

        [Tooltip("The area in which to place objects in the 2D plane.")]
        public Vector2 placementArea = new Vector2(6f, 6f);

        [Tooltip("The prefabs to sample from for object generation.")]
        public CategoricalParameter<GameObject> prefabs;

        [Tooltip("If true, the color of the object will be adjusted.")]
        public bool adjustColor = false;

        [Tooltip("The materials to sample from for object color adjustment. (If Texture Randomizer is used, this will be overridden.)")]
        public CategoricalParameter<Material> materials;

        GameObject m_Container;
        GameObjectOneWayCache m_GameObjectOneWayCache;
    
        protected override void OnAwake()
        {
            m_Container = new GameObject("Foreground Occluders");
            var transform = scenario.transform;
            m_Container.transform.parent = transform;
            m_GameObjectOneWayCache = new GameObjectOneWayCache(
                m_Container.transform, prefabs.categories.Select(element => element.Item1).ToArray(), this);
        }

        /// <summary>
        /// Generates a foreground layer of objects at the start of each scenario iteration
        /// </summary>
        protected override void OnIterationStart()
        {
            var seed = SamplerState.NextRandomState();

            var placementSamples = PoissonDiskSampling.GenerateSamples(
                placementArea.x, placementArea.y, occluderSeparationDistance.Sample(), seed);
            var offset = new Vector3(placementArea.x, placementArea.y, 0f) * -0.5f;
        
            foreach (var sample in placementSamples)
            {
                var prefab = prefabs.Sample();
                var scaleFactor = scale.Sample();
                var depthOffset = depth.Sample();
                var colorMaterial = materials.Sample();

                var instance = m_GameObjectOneWayCache.GetOrInstantiate(prefab);
                instance.transform.position = new Vector3(sample.x, sample.y, depthOffset) + offset;
                instance.transform.localScale = new Vector3(scaleFactor, scaleFactor, scaleFactor);
                if (adjustColor)
                {
                    var renderer = instance.GetComponent<Renderer>();
                    if (renderer != null)
                    {
                        renderer.material = colorMaterial;
                    }
                }
            }
            
            placementSamples.Dispose();
        }
        
        /// <summary>
        /// Deletes generated foreground objects after each scenario iteration is complete
        /// </summary>
        protected override void OnIterationEnd()
        {
            m_GameObjectOneWayCache.ResetAllObjects();
        }
    }
}

