using System;
using System.Linq;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;
using UnityEngine.Perception.Randomization.Utilities;
using LightRandomizerTag = SynthDet.RandomizerTags.LightRandomizerTag;

namespace SynthDet.Randomizers
{
    [Serializable]
    [AddRandomizerMenu("SynthDet/Light Randomizer")]
    public class LightRandomizer : Randomizer
    {
        [Header("Randomization Settings")]
        [Tooltip("Whether the light properties should be adjusted")]
        public bool adjustLightProperties = false;
        [Tooltip("Whether the light should be rotated and positioned")]
        public bool adjustRotationAndPosition = true;
        [Tooltip("Whether the light should cast shadows")]
        public bool castShadows = false;
        [Tooltip("Whether the light should cast soft shadows")]
        public bool softShadows = true;
        [Tooltip("The scale of the distractor objects")]
        public float occluderScale = 20.0f;
        [Tooltip("Distractor objects for casting shadows")]
        public CategoricalParameter<GameObject> prefabsForShadows; 


        [Header("Light Parameters")]
        [Tooltip("The intensity / brightness of the light")]
        public FloatParameter lightIntensityParameter = new() { value = new UniformSampler(0f, 1f) };

        [Tooltip("The color of the light")]
        public ColorRgbParameter lightColorParameter = new()
        {
            red = new UniformSampler(1.0f, 1f),
            green = new UniformSampler(1.0f, 1f),
            blue = new UniformSampler(1.0f, 1f),
            alpha = new ConstantSampler(1f)
        };

        [Header("Position and Rotation Parameters")]
        [Tooltip("The distribution of the light's rotation in degrees")]
        // Make a rotation parameter in all axes
        public Vector3Parameter rotationDistribution = new()
        {
            x = new UniformSampler(0f, 0f),
            y = new UniformSampler(180f, 180f),
            z = new UniformSampler(0, 0)
        };

        [Tooltip("The distribution of the light's position")]
        // Make a position parameter in all axes
        public Vector3Parameter positionDistribution = new()
        {
            x = new UniformSampler(-2f, 2f),
            y = new ConstantSampler(-4f),
            z = new ConstantSampler(2f)
        };

        public bool adjustTest = false;

        [Header("Test Parameters")]

        [Tooltip("The number of iterations that are used for the testing split. If set to -1, the test split is half of the total iterations.")]
        public int numTrainImages = -1;

        [Tooltip("Rotates the light position around the x-axis from -testRotationXRange/2 to testRotationXRange/2. This roation is from top to bottom.")]
        public float testRotationXRange = 0f;
        [Tooltip("Rotates the light position around the y-axis from -testRotationYRange/2 to testRotationYRange/2. This roation is from left to right.")]
        public float testRotationYRange = 90f;
        [Tooltip("Rotates the light position around the z-axis from -testRotationZRange/2 to testRotationZRange/2. This roation is from front to back.")]
        public float testRotationZRange = 0f;

        GameObject m_Container;
        GameObjectOneWayCache m_GameObjectOneWayCache;

        private int iterationCount = 1;
        private Vector3 m_InitialLightRotation = new Vector3(0, 180, 0);

        protected override void OnScenarioStart() 
        {
            var constants = scenario.GetType().GetField("constants").GetValue(scenario);
            iterationCount = (int)constants.GetType().GetField("iterationCount").GetValue(constants);

            if (numTrainImages == -1)
            {
                numTrainImages = (int)(iterationCount / 2);
            }
        }

        protected override void OnAwake()
        {
            m_Container = new GameObject("Background Distractors");
            m_Container.transform.parent = scenario.transform;
            m_GameObjectOneWayCache = new GameObjectOneWayCache(m_Container.transform,
                prefabsForShadows.categories.Select((element) => element.Item1).ToArray(), this);
        }

        protected override void OnIterationStart()
        {
            var seed = SamplerState.NextRandomState();
            UnityEngine.Random.InitState((int)seed);
            
            var randomizerTags = tagManager.Query<LightRandomizerTag>();
            foreach (var tag in randomizerTags)
            {
                var light = tag.GetComponent<Light>();
                if (adjustLightProperties)
                {
                    light.color = lightColorParameter.Sample();
                    tag.SetIntensity(lightIntensityParameter.Sample());
                }
                if (adjustRotationAndPosition)
                {
                    var rotation = rotationDistribution.Sample();
                    if (adjustTest && (int)scenario.currentIteration >= numTrainImages)
                    {
                        var percent = (float)(scenario.currentIteration - numTrainImages) / (float)(iterationCount - numTrainImages);
                        percent = percent - 0.5f;
                        var lightRotation = m_InitialLightRotation;
                        var x = lightRotation.x + testRotationXRange * percent;
                        var y = lightRotation.y + testRotationYRange * percent;
                        var z = lightRotation.z + testRotationZRange * percent;
                        rotation = new Vector3(x, y, z);
                    }

                    tag.transform.rotation = Quaternion.Euler(rotation);
                    var segmentationObjectsPosition = Vector3.zero;
                    light.transform.position = positionDistribution.Sample();
                    var newPosition = RePositionLightObject(light, segmentationObjectsPosition);
                    tag.transform.position = newPosition;

                }
                if (castShadows)
                {
                    if (softShadows)
                    {
                        light.shadows = LightShadows.Soft;
                    } 
                    InstantiateOccludingObjects(light);
                }
            }
        }

        protected override void OnIterationEnd()
        {
            m_GameObjectOneWayCache.ResetAllObjects();
            var randomizerTags = tagManager.Query<LightRandomizerTag>();
        }

        Vector3 RePositionLightObject(Light light, Vector3 focusPoint)
        {
            // Get the distance between light.transform.position and focusPoint
            var distance = Vector3.Distance(light.transform.position, focusPoint);
            // Get the direction from light.transform.position to focusPoint
            var direction = light.transform.forward;
            // Set the new position of the light object
            return focusPoint - direction * distance;
        }

        void InstantiateOccludingObjects(Light light, int occluderCount = 10, float occluderDistance = 5f)
        {
            // Instantiate occluding objects
            for (var i = 0; i < occluderCount; i++)
            {
                var prefab = prefabsForShadows.Sample();
                var instance = m_GameObjectOneWayCache.GetOrInstantiate(prefab);
                // Get the direction of the light
                var lightDirection = light.transform.forward;
                // Get a random point in the direction of the light shifted by the occluder distance
                var occluderPoint = light.transform.position + lightDirection * occluderDistance;
                // Create a random position for an occluder object in the plane perpendicular to the light direction
                // Generate random offsets in x and y directions perpendicular to the light direction
                var randomOffsetX = UnityEngine.Random.Range(-1f, 1f); // Adjust the range as needed
                var randomOffsetY = UnityEngine.Random.Range(-1f, 1f); // Adjust the range as needed

                // Calculate a vector perpendicular to the light direction for x and y offsets
                var perpendicularX = Vector3.Cross(lightDirection, Vector3.up).normalized;
                var perpendicularY = Vector3.Cross(lightDirection, perpendicularX).normalized;

                // Calculate the final position of the occluder object
                var finalPosition = occluderPoint + perpendicularX * randomOffsetX + perpendicularY * randomOffsetY;

                // Set the position of the occluder object
                instance.transform.position = finalPosition;
                // Set the scale of the occluder object
                instance.transform.localScale = new Vector3(occluderScale, occluderScale, occluderScale);
            }

        }
    }
}
