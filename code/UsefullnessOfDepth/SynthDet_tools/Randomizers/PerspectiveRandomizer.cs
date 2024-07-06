using System;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;
using CameraRandomizerTag = SynthDet.RandomizerTags.CameraRandomizerTag;
using LightRandomizerTag = SynthDet.RandomizerTags.LightRandomizerTag;

namespace SynthDet.Randomizers
{
    [Serializable]
    [AddRandomizerMenu("SynthDet/Perspective Randomizer")]
    public class PerspectiveRandomizer : Randomizer
    {
        [Tooltip("Configures if the entire image is rotated along z axis")]
        public float maxZrotation = 0;

        [Tooltip("Configures if the camera position is changed along the y axis and angle along the x axis (up and down)")]
        public bool trainCubicZRotations = false;

        [Tooltip("Configures if the camera position is changed along the y axis and angle along the x axis (up and down)")]
        public Vector2 heightOffsetRange = new Vector2(0f, 0f);

        [Tooltip("Configures if the camera position is changed along the x axis and angle along the y axis (left and right)")]
        public Vector2 widthOffsetRange = new Vector2(0f, 0f);

        [Tooltip("Configures if the light rotation is adjusted accordingly to the camera rotation")]
        public bool adjustLightingRotationAccordingly = false;

        [Range(90f, 110f)]
        public float cameraDistance = 100;

        private Camera mainCamera;

        private int iteration_count = 1;

        protected override void OnScenarioStart() 
        {
            var constants = scenario.GetType().GetField("constants").GetValue(scenario);
            iteration_count = (int)constants.GetType().GetField("iterationCount").GetValue(constants);
        }

        protected override void OnIterationStart()
        {
            var seed = SamplerState.NextRandomState();
            UnityEngine.Random.InitState((int)seed);

            var angleOffsetX = 0f;
            var angleOffsetY = 0f;
            
            var cameraTags = tagManager.Query<CameraRandomizerTag>();
            foreach (var tag in cameraTags)
            {
                var camera = tag.GetComponent<Camera>();
                mainCamera = camera;

                // Rotate the entire image along z axis
                if (maxZrotation != 0 && !trainCubicZRotations)
                {
                    var zRotation = UnityEngine.Random.Range(-maxZrotation, maxZrotation);
                    tag.transform.Rotate(0, 0, zRotation);
                }
                if (trainCubicZRotations && (int)scenario.currentIteration < (iteration_count / 2))
                {
                    var zRotation = UnityEngine.Random.Range(0, 4) * 90;
                    tag.transform.Rotate(0, 0, zRotation);
                }

                // Change the camera position along the y axis and angle along the x axis (up and down)
                if (heightOffsetRange.x != 0 && heightOffsetRange.y != 0)
                {
                    var heightOffset = UnityEngine.Random.Range(heightOffsetRange.x, heightOffsetRange.y);
                    var distance = 100;
                    angleOffsetX = Mathf.Atan(heightOffset / distance) * Mathf.Rad2Deg;
                    tag.transform.position += new Vector3(0, heightOffset, 0);
                    tag.transform.rotation = Quaternion.Euler(
                        tag.transform.rotation.eulerAngles.x + angleOffsetX, 
                        tag.transform.rotation.eulerAngles.y, 
                        tag.transform.rotation.eulerAngles.z   
                    );
                }

                // Change the camera position along the x axis and angle along the y axis (left and right)
                if (widthOffsetRange.x != 0 && widthOffsetRange.y != 0)
                {
                    var widthOffset = UnityEngine.Random.Range(widthOffsetRange.x, widthOffsetRange.y);
                    var distance = 100;
                    angleOffsetY = Mathf.Atan(widthOffset / distance) * Mathf.Rad2Deg;
                    tag.transform.position += new Vector3(widthOffset, 0, 0);
                    tag.transform.rotation = Quaternion.Euler(
                        tag.transform.rotation.eulerAngles.x, 
                        tag.transform.rotation.eulerAngles.y + angleOffsetY, 
                        tag.transform.rotation.eulerAngles.z   
                    );
                }

                // Move the camera in the direction which it is facing, so that the distance between the point of interest(0,0,0) and the camera 100
                tag.transform.position = -tag.transform.forward.normalized * cameraDistance;
            }

            if (adjustLightingRotationAccordingly)
            {
                var lightTags = tagManager.Query<LightRandomizerTag>();
                foreach (var tag in lightTags)
                {
                    var light = tag.GetComponent<Light>();
                    light.transform.rotation = Quaternion.Euler(
                        light.transform.rotation.eulerAngles.x + angleOffsetX, 
                        light.transform.rotation.eulerAngles.y + angleOffsetY, 
                        light.transform.rotation.eulerAngles.z
                    );
                }
            }

        }

        protected override void OnIterationEnd()
        {
            var randomizerTags = tagManager.Query<CameraRandomizerTag>();
            foreach (var tag in randomizerTags)
            {
                var camera = tag.GetComponent<Camera>();
                mainCamera = camera;

                // Reset the entire image rotation along z axis
                mainCamera.transform.rotation = Quaternion.Euler(0, 180, 0);
                mainCamera.transform.position = new Vector3(0, 0, 100);
            }

            var lightTags = tagManager.Query<LightRandomizerTag>();
            foreach (var tag in lightTags)
            {
                var light = tag.GetComponent<Light>();
                light.transform.rotation = Quaternion.Euler(0, 180, 0);
            }
        }
    }
}
