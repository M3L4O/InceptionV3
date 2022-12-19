from sklearn.model_selection import train_test_split
import imgaug.augmenters as iaa
import imgaug as ia
import pandas as pd
import numpy as np
import os
import cv2


def get_clean_df(filename, input_dir):
    """
    Função utilizada para remover as colunas não utilizadas e padronizar o DataFrame.

    Parametros
    ----------
        filename : str
            Caminho até o arquivo contendo os metadados das instâncias.
        input_dir : str
            Caminho até as imagens.

    Returns
    -------
    DataFrame
        DataFrame contendo as informações necessárias para o treinamento.

    """
    df = pd.read_csv(filename)
    df = df[["image_name", "benign_malignant"]]
    columns = ["image", "label"]
    df.columns = columns
    df["image"] = df["image"].apply(lambda x: f"{input_dir}/{x}.jpg")
    return df


def get_augmenter():
    """
    Função utilizada para gerar um augmenter.

    Returns
    -------
    imgaug.augmenters.meta.SomeOf
        Função necessárias para ampliar a base de imagens.

    """
    ia.seed(123)

    rotations = iaa.OneOf(
        [
            iaa.Rotate((60, 300)),
            iaa.Rotate((45, 315)),
            iaa.Rotate((90, 270)),
        ]
    )

    zoom = iaa.OneOf(
        [
            iaa.Affine(scale=(1.75, 2.0)),
            iaa.Affine(scale=(1.0, 1.5)),
            iaa.Affine(scale=(1.75, 2.0)),
        ]
    )

    contrast = iaa.OneOf(
        [
            iaa.GammaContrast((0.5, 2.0)),
            iaa.LogContrast(gain=(0.6, 1.4)),
            iaa.LinearContrast((0.4, 1.6)),
        ]
    )

    geometric = iaa.SomeOf(
        2,
        [
            iaa.Affine(shear=(-16, 16)),
            iaa.ShearX((-20, 20)),
            iaa.ShearY((-20, 20)),
        ],
    )

    noise = iaa.OneOf(
        [
            iaa.imgcorruptlike.GaussianNoise(severity=2),
            iaa.imgcorruptlike.ShotNoise(severity=2),
            iaa.imgcorruptlike.ImpulseNoise(severity=2),
            iaa.imgcorruptlike.SpeckleNoise(severity=2),
        ]
    )

    aug = iaa.Sequential(
        [
            iaa.Sometimes(0.3, noise),
            zoom,
            rotations,
            iaa.SomeOf(
                (1, None),
                [
                    contrast,
                    geometric,
                    noise,
                ],
            ),
        ]
    )

    return aug


def augment_generator(
    df, augmenter, dir_aug_images, img_size=(224, 224), new_images=10
):
    """
    Função utilizada para ampliar a base de imagens.

    Parametros
    ----------
        df : pd.DataFrame
            DataFrame contendo as instância.

        augmenter : imgaug.augmenters.meta.SomeOf
            Objeto responsável por aumentar a quantidade de versões de uma imagem.

        dir_aug_images : str
            Caminho até o diretório que irá armazenar as imagens geradas.

        img_size : tuple
            Tamanho da imagem

        new_images : int
            Número de novas imagens geradas a cada imagem.

    Returns
    -------
    DataFrame
        DataFrame contendo as informações das novas imagens geradas.

    """
    print("Progresso: ")
    new_rows = []
    for i in range(len(df)):
        filename = os.path.basename(df["image"].iloc[i]).split(".")[0]
        label = df["label"].iloc[i]
        image = cv2.imread(df["image"].iloc[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        aug_images = augmenter(images=[image] * new_images)

        for j in range(new_images):
            new_filepath = os.path.join(dir_aug_images, f"{filename}_{j}.jpg")
            cv2.imwrite(new_filepath, cv2.resize(cv2.cvtColor(aug_images[j], cv2.COLOR_RGB2BGR), img_size))
            new_rows.append((new_filepath, label))

        print(f"{i+1}/{len(df)}")

    aug_df = pd.DataFrame(new_rows, columns=["image", "label"])

    return aug_df


def split_dataframe(df, train_file, validation_file, test_file, augment_dir):
    """
    Função utilizada para separar dividir o Dataset
    Parametros
    ----------
        df : pd.DataFrame
            DataFrame contendo as instâncias.

        train_file: str
            Nome do arquivo que armazenará as instâncias de treino.

        validation_file: str
            Nome do arquivo que armazenará as instâncias de validação.

        test_file: str
            Nome do arquivo que armazenará as instâncias de teste.

    """
    malignant_df = df.loc[df["label"] == "malignant"]
    benign_df = df.loc[df["label"] == "benign"].sample(malignant_df.shape[0])

    balanced_df = pd.concat([malignant_df, benign_df], ignore_index=True)
    df = pd.concat([df, balanced_df]).drop_duplicates(keep=False)

    train_df, dummy_df = train_test_split(
        balanced_df, train_size=0.8, shuffle=True, random_state=1234
    )

    validation_df, test_df = train_test_split(
        dummy_df, train_size=0.5, shuffle=False, random_state=1234
    )

    if not os.path.exists(augment_dir):
        os.mkdir(augment_dir)

    validation_df.to_csv(validation_file)
    test_df.to_csv(test_file)

    malignant_train_df = train_df.loc[train_df["label"] == "malignant"]
    benign_train_df = train_df.loc[train_df["label"] == "benign"]

    new_images = int(
        (df.shape[0] + benign_train_df.shape[0]) / malignant_train_df.shape[0]
    )

    aug_df = augment_generator(
        malignant_train_df, get_augmenter(), augment_dir, new_images=new_images
    )

    train_df = pd.concat([aug_df, df, train_df], ignore_index=True)

    train_df.to_csv(train_file)


def main():

    metadata_filepath = r"D:\Bases de Imagens\ISIC2020\Meta_ISIC\ISIC_2020_Training_GroundTruth.csv"
    input_dir = r"D:\Bases de Imagens\ISIC2020\ISIC_2020_train"

    train_file = "train.csv"
    validation_file = "validation.csv"
    test_file = "test.csv"

    augment_dir = r"D:\Bases de Imagens\ISIC2020\aug_train_malignant_images"
    df = get_clean_df(metadata_filepath, input_dir)
    split_dataframe(df, train_file, validation_file, test_file, augment_dir)


if __name__ == "__main__":
    main()

