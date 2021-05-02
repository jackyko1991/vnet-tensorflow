import NiftiDataset2D,NiftiDataset3D
import yaml


def main():
	# load the yaml
	with open("pipeline2D.yaml") as f:
		pipeline_ = yaml.load(f)

	# start preprocessing
	print(pipeline_["preprocess"])

	train_transform_3d = []
	train_transform_2d = []
	test_transform_3d = []
	test_transform_2d = []

	if pipeline_["preprocess"]["train"]["3D"] is not None:
		for transform in pipeline_["preprocess"]["train"]["3D"]:
			tfm_cls = getattr(NiftiDataset3D,transform["name"])(*[],**transform["variables"])
			train_transform_3d.append(tfm_cls)

	if pipeline_["preprocess"]["train"]["2D"] is not None:
		for transform in pipeline_["preprocess"]["train"]["2D"]:
			tfm_cls = getattr(NiftiDataset2D,transform["name"])(*[],**transform["variables"])
			train_transform_2d.append(tfm_cls)

	if pipeline_["preprocess"]["test"]["3D"] is not None:
		for transform in pipeline_["preprocess"]["test"]["3D"]:
			tfm_cls = getattr(NiftiDataset3D,transform["name"])(*[],**transform["variables"])
			test_transform_3d.append(tfm_cls)

	if pipeline_["preprocess"]["test"]["2D"] is not None:
		for transform in pipeline_["preprocess"]["test"]["2D"]:
			tfm_cls = getattr(NiftiDataset2D,transform["name"])(*[],**transform["variables"])
			test_transform_2d.append(tfm_cls)

	print(train_transform_3d.__dict__)
	print(train_transform_2d.__dict__)
	print(test_transform_3d.__dict__)
	print(test_transform_2d.__dict__)

if __name__ == "__main__":
	main()