using System;
using SynthDet.RandomizerTags;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;

namespace SynthDet.Randomizers
{
    [Serializable]
    [AddRandomizerMenu("SynthDet/Foreground Scale Randomizer For Non-Normalized Meshes")]
    public class ForegroundScaleRandomizerForNonNormalizedMeshes : Randomizer
    {
        public FloatParameter scale = new FloatParameter { value = new UniformSampler(4f, 8f) };

		public bool normalizeObjectAxes = false;

		public bool normalizeObjectScale = true;

		private Bounds GetMeshBounds(ForegroundScaleRandomizerTag tag)
		{
			// Get the object's mesh
			MeshFilter meshFilter = tag.GetComponent<MeshFilter>();
			if (meshFilter == null)
			{
				foreach (Transform child in tag.transform)
				{
					meshFilter = child.GetComponent<MeshFilter>();
					if (meshFilter != null)
					{
						break;
					}
				}
				if (meshFilter == null)
				{
					return new Bounds(new Vector3(0, 0, 0), new Vector3(0, 0, 0));
				}
			}

			Mesh mesh = meshFilter.sharedMesh;
			if (mesh == null)
			{
				return new Bounds(new Vector3(0, 0, 0), new Vector3(0, 0, 0));
			}

			// Calculate the bounds of the mesh
			Bounds bounds = mesh.bounds;

			return bounds;
		}

        protected override void OnIterationStart()
        {
			var largestBounds = 0.0f;

            var tags = tagManager.Query<ForegroundScaleRandomizerTag>();
            foreach (var tag in tags)
            {
				// Get the object's mesh
				Bounds bounds = GetMeshBounds(tag);
				 
				float maxBounds = Mathf.Max(bounds.size.x, bounds.size.y, bounds.size.z);
				if (maxBounds > largestBounds) {
					largestBounds = maxBounds;
				}
            }

			// Normalize the scale of all objects with the largest bounds
			foreach (var tag in tags)
			{
				// Get the object's mesh
				Bounds bounds = GetMeshBounds(tag);

				if (normalizeObjectAxes) {
					// Normalize the scale of the object
					float maxBounds = Mathf.Max(bounds.size.x, bounds.size.y, bounds.size.z);
					Vector3 ScaledBounds = new Vector3(
						1.0f / (bounds.size.x / maxBounds),
						1.0f / (bounds.size.y / maxBounds),
						1.0f / (bounds.size.z / maxBounds)
					);

					tag.transform.localScale = Vector3.Normalize(ScaledBounds);
				}

				// Scale the object
				tag.transform.localScale *= scale.Sample();

				if (normalizeObjectScale) {
					// Normalize the scale of the object
					float avgBounds = (bounds.size.x + bounds.size.y + bounds.size.z) / 3.0f;
					// float minBounds = Mathf.Min(bounds.size.x, bounds.size.y, bounds.size.z);
					// float maxBounds = Mathf.Max(bounds.size.x, bounds.size.y, bounds.size.z);
					float scalingFactor = largestBounds / avgBounds;

					Vector3 normalizedScale = new Vector3(
						tag.transform.localScale.x * scalingFactor,
						tag.transform.localScale.y * scalingFactor,
						tag.transform.localScale.z * scalingFactor
					);

					tag.transform.localScale = normalizedScale;
				}
			}
        }
    }
}