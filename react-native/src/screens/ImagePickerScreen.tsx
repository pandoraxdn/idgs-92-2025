import React, { useState, useEffect } from 'react';
import { View, Text, Alert, Button, Image } from 'react-native';
import { appTheme } from '../themes/appTheme';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';

export const ImagePickerScreen = () => {

    const [ image, setImage ] = useState<string|null>(null);
    const [ image64, setImage64 ] = useState<string|null>(null);

    useEffect( () => {
        ( async () => {
            const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
            if ( status !== "granted" ){
                Alert.alert(
                    "Permiso requerido",
                    "Debes otorgar los permisos para acceder a la galería."
                ); 
            }
        })();
    },[]);

    const pickImage = async () => {
        let result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ["images"],
            allowsEditing: true,
            aspect: [4,3],
            quality: 0.9,
        });

        ( !result.canceled ) && ( () => {
            setImage( result.assets[0].uri );
            convertImage64( result.assets[0].uri );
        })();
    }

    const convertImage64 = async ( imageUri: string ) => {
        try{
            const base64 = await FileSystem.readAsStringAsync( imageUri,{
                encoding: FileSystem.EncodingType.Base64,
            });
            setImage64( base64 );
        }catch (error){
            console.log(error);
        }
    }

    return(
        <View
            style={ appTheme.container }
        >
            <Text
                style={ appTheme.text }
            >
                Selección de Imagen
            </Text>
            <Button
                title='Seleccionar Imagen'
                onPress={ () => pickImage() }
            /> 
            {
                (image) && (
                    <Image 
                        source={{
                            uri: `data:image/jpeg;base64,${image64}`
                        }}
                        style={{
                            width: 200,
                            height: 300,
                            borderRadius: 20
                        }}
                    />
                )
            }
        </View>
    )
}
