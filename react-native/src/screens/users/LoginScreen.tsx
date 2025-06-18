import React, { useRef, useEffect } from 'react';
import { View, Text, TextInput, StyleSheet, Animated, Dimensions, ImageBackground } from 'react-native';
import { BtnTouch } from '../../components/BtnTouch';

export const LoginScreen = () => {

    const { width, height } = Dimensions.get("window");

    const plumaImage = require('./../../../assets/pluma.png');

    const NUMERO_PLUMAS: number = 40;

    const plumas = Array.from({ length: NUMERO_PLUMAS }, () => ({
        animation: useRef( new Animated.Value(0)).current,
        startX: Math.random() * width,
        driftX: Math.random() * 60 - 30,
        rotation: `${Math.random()*360}deg`
    }));

    useEffect( () => {
        const interval = setInterval(() => {
            plumas.forEach((pluma) => {
                const animatePluma = () => {
                    pluma.animation.setValue(0);
                    Animated.timing( pluma.animation, {
                        toValue: 1,
                        duration: 6000 + Math.random() * 4000,
                        useNativeDriver: true,
                    }).start( () => animatePluma );
                }
                animatePluma();
            });
            return () => clearInterval(interval);
        },7000);
    },[]);

    return(
        <ImageBackground
            source={require('./../../../assets/image.jpeg')}
            style={ style.container }
            resizeMode='cover'
        >
            {plumas.map( (pluma, index) => {

                    const translateY = pluma.animation.interpolate({
                        inputRange: [0,1],
                        outputRange: [-50, height + 50],
                    });

                    const translateX = pluma.animation.interpolate({
                        inputRange: [0,1],
                        outputRange: [pluma.startX, pluma.driftX + pluma.startX],
                    });

                    const rotate = pluma.animation.interpolate({
                        inputRange: [0,1],
                        outputRange: ['0deg', pluma.rotation]
                    });

                    return(
                        <Animated.Image
                            key={ index }
                            source={ plumaImage }
                            style={[
                                style.pluma,
                                {
                                    transform: [{ translateY },{translateX},{rotate}]
                                },
                            ]}
                        />
                    )
            })}

            {/*Formulario*/}
            <View
                style={ style.loginBox }
            >
                <Text
                    style={ style.title }
                >
                    Login
                </Text>
                <TextInput
                    style={ style.input }
                    placeholder='username'
                    placeholderTextColor="#ccc"
                />
                <TextInput
                    style={ style.input }
                    placeholder='contraseña'
                    placeholderTextColor="#ccc"
                    secureTextEntry={true}
                />
                <View
                    style={{
                        justifyContent: "center",
                        alignItems: "center"
                    }}
                >
                    <BtnTouch
                        title="Ingresar"
                        background='black'
                        bColor='white'
                        onPress={ () => console.log("Algo") }
                    />
                </View>
            </View>
        </ImageBackground>
    )
}

const style = StyleSheet.create({
    container:{
        flex: 1,
    },
    pluma:{
        position: "absolute",
        width: 100,
        height: 100,
        opacity: 0.6
    },
    loginBox:{
        position: "absolute",
        bottom: 100,
        width: "90%",
        alignSelf: "center",
        backgroundColor: "rgba(255,255,255,0.9)",
        borderRadius: 20,
        padding: 20
    },
    title:{
        color: "black",
        fontSize: 30,
        marginBottom: 20,
        textAlign: "center"
    },
    input: {
        backgroundColor: "#222",
        color: "#fff",
        borderRadius: 10,
        padding: 12,
        marginBottom: 15,
    }
});
