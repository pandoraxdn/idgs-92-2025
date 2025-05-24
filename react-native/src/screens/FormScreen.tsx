import React from 'react';
import { View, Text, TextInput, TouchableOpacity, ScrollView } from 'react-native';
import { useFormHook } from '../hooks/useFormData';
import { appTheme } from '../themes/appTheme';

export const FormScreen = () => {

    const { formData, formList, handleInputChange, handleSubmit } = useFormHook();

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
                <View>
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
                    <Text
                        style={ appTheme.subtext }
                    >
                        Edad
                    </Text>
                    <TextInput
                        style={ appTheme.input }
                        value={ formData.edad }
                        onChangeText={ (text) => handleInputChange('edad',text) }
                        placeholder='Nombre del usuario'
                        placeholderTextColor='violet'
                        keyboardType='numeric'
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
