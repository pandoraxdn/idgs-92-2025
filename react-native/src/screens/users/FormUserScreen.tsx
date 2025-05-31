import React, { useEffect } from 'react';
import { View, Text, TextInput, TouchableOpacity, ScrollView } from 'react-native';
import { RootStackUserParams } from '../../navigator/UserNavigator';
import { StackScreenProps } from '@react-navigation/stack';
import { useFormHook } from '../../hooks/useFormData';
import { appTheme } from '../../themes/appTheme';
import { BtnTouch } from '../../components/BtnTouch';

interface Props extends StackScreenProps<RootStackUserParams,'FormScreen'>{};

export const FormScreen = ( { navigation, route }: Props ) => {

    const { formData, formList, handleInputChange, handleSubmit } = useFormHook();
    
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
                            title='Eliminar Usuario'
                            background='red'
                            onPress={ () => {
                                console.log("eliminar")
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
                    <TouchableOpacity
                        style={ appTheme.btn }
                        onPress={ () => handleSubmit() }
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
                </View>
            </View>
        </ScrollView>
    )
}
