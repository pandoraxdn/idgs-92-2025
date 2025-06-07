import React, { useEffect, useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, ScrollView, Alert, Image } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';
import { RootStackUserParams } from '../../navigator/UserNavigator';
import { StackScreenProps } from '@react-navigation/stack';
import { useFormHook } from '../../hooks/useFormData';
import { appTheme } from '../../themes/appTheme';
import { BtnTouch } from '../../components/BtnTouch';

interface Props extends StackScreenProps<RootStackUserParams,'FormScreen'>{};

interface BtnForm{
    action: () => void;
    estado: string;
    tipo: string;
}

const BtnForm = ( { tipo, action, estado } : BtnForm ) => {

    let colorBtn: string = "white";

    switch( tipo ){
        case 'client':
            colorBtn = "#e8c70b";
            break;
        case 'user':
            colorBtn = "#90EE90";
            break;
        case 'admin':
            colorBtn = "#d90000";
            break;
    }
    
    return(
        <TouchableOpacity
            onPress={ action }
        >
            <View
                style={{
                    backgroundColor: ( estado === '' ) ? "white": colorBtn,
                    borderRadius: 30,
                    borderWidth : (estado === '') ? 1 : 0,
                    marginHorizontal: 5,
                    justifyContent: "center",
                    height: 60,
                    width: 90,
                    marginTop: 10,
                    marginBottom: 10
                }}
            >
                <Text
                    style={{
                        color: (estado === '') ? 'black': 'white',
                        fontWeight: "bold",
                        textAlign: "center",
                        fontSize: 17
                    }}
                >
                    { tipo }
                </Text>
            </View>
        </TouchableOpacity>
    );
}

export const FormScreen = ( { navigation, route }: Props ) => {

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
            handleInputChange('imagen',base64);
        }catch (error){
            console.log(error);
        }
    }

    const { formData, handleInputChange, handleSubmit, handleDelete } = useFormHook();
    
    const user = route.params.user;
    
    useEffect( () => {
        handleInputChange('_id',user._id);
        handleInputChange('username',user.username);
        handleInputChange('imagen',user.imagen);
        handleInputChange('tipo',user.tipo);
    },[]);

    return(
        <ScrollView
            style={{
                marginTop: 12
            }}
        >
            <View
                style={ appTheme.container }
            >
                <Text
                    style={ appTheme.text }
                >
                    Formulario de Usuarios
                </Text>
                <View
                    style={{
                        alignContent: "center",
                        alignItems: "center"
                    }}
                >
                    {
                        (formData._id != "") &&
                        <BtnTouch
                            title={`Eliminar User:`}
                            background='red'
                            onPress={ () => {
                                handleDelete();
                                navigation.goBack();
                            }}
                            bColor='pink'
                        />
                    }
                    <Text
                        style={ appTheme.subtext }
                    >
                        Username
                    </Text>
                    <TextInput
                        style={ appTheme.input }
                        value={ formData.username }
                        onChangeText={ (text) => handleInputChange('username',text) }
                        placeholder='Nombre del usuario'
                        placeholderTextColor='violet'
                    />
                    <Text
                        style={ appTheme.subtext }
                    >
                        Password
                    </Text>
                    <TextInput
                        style={ appTheme.input }
                        value={ formData.password }
                        onChangeText={ (text) => handleInputChange('password',text) }
                        placeholder='Nombre del usuario'
                        placeholderTextColor='violet'
                        secureTextEntry={ true }
                    />
                    <View>
                        <BtnTouch
                            title='Avatar'
                            onPress={() => pickImage() }
                            background='black'
                            bColor='gray'
                        />
                        {
                            (formData.imagen) && (
                                <Image 
                                    source={{
                                        uri: `data:image/jpeg;base64,${formData.imagen}`
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
                    <View
                        style={{ flexDirection: "row", marginTop: 5 }}
                    >
                        <BtnForm
                            tipo='client'
                            action={ () => handleInputChange('tipo','client') }
                            estado={ ( formData.tipo === "client" ) ? formData.tipo : '' }
                        />
                        <BtnForm
                            tipo='admin'
                            action={ () => handleInputChange('tipo','admin') }
                            estado={ ( formData.tipo === "admin" ) ? formData.tipo : '' }
                        />
                        <BtnForm
                            tipo='user'
                            action={ () => handleInputChange('tipo','user') }
                            estado={ ( formData.tipo === "user" ) ? formData.tipo : '' }
                        />
                    </View>
                    <TouchableOpacity
                        style={ appTheme.btn }
                        onPress={ () => {
                            handleSubmit();
                            navigation.navigate("HomeUserScreen");
                        }}
                    >
                        <View
                            style={{
                                width: 100,
                                height: 60,
                                backgroundColor: "violet",
                                borderRadius: 20,
                                alignContent: "center",
                                alignItems: "center",
                                justifyContent: "center",
                                borderWidth: 5,
                                borderColor: "pink"
                            }}
                        >
                            <Text
                                style={{
                                    fontSize: 30,
                                    color: "white",
                                    alignSelf: "center"
                                }}
                            >
                                Enviar
                            </Text>
                        </View>
                    </TouchableOpacity>
                    {/*
                        <View>
                            {( formList.map( (form, index) => (
                                <Text
                                    key={ index }
                                    style={{ fontSize: 16 }}
                                >
                                    { JSON.stringify(form) }
                                </Text>
                            )))}
                        </View>
                    */}
                </View>
            </View>
        </ScrollView>
    )
}
