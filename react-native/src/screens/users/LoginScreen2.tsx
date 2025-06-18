import React, { useRef, useEffect } from 'react';
import { View, Text, TextInput, StyleSheet, Animated, Dimensions, ImageBackground } from 'react-native';
import { BtnTouch } from '../../components/BtnTouch';
import { useLoginUser } from '../../hooks/useLoginUser';

export const LoginScreen2 = () => {

    const { loading, handleLogin, handleInputChange, request, state } = useLoginUser();

    const { width, height } = Dimensions.get("window");

    const burbujaImage = require('./../../../assets/burbuja.png');

    const NUMERO_BURBUJAS: number = 40;

    const burbujas = Array.from({ length: NUMERO_BURBUJAS }, () => ({
        animation: useRef( new Animated.Value(0)).current,
        startX: Math.random() * width,
        driftX: Math.random() * 60 - 30,
        scale: 0.4 + Math.random() * 0.6,
        delay: Math.random() * 3000
    }));

    useEffect( () => {
        const interval = setInterval(() => {
            burbujas.forEach((burbuja) => {
                const animateBurbuja = () => {
                    burbuja.animation.setValue(0);
                    Animated.timing( burbuja.animation, {
                        toValue: 1,
                        duration: 7000 + Math.random() * 3000,
                        delay: burbuja.delay,
                        useNativeDriver: true
                    }).start( () => animateBurbuja );
                }
                animateBurbuja();
            });
            return () => clearInterval(interval);
        },7000);
    },[]);
    return(
        <ImageBackground
            source={require("./../../../assets/image.jpeg")}
            style={ style.container }
        >
            { burbujas.map((burbuja,index) => {
                
                    const translateY = burbuja.animation.interpolate({
                        inputRange: [0,1],
                        outputRange: [50 + height, -100],
                    });

                    const translateX = burbuja.animation.interpolate({
                        inputRange: [0,1],
                        outputRange: [burbuja.startX, burbuja.driftX + burbuja.startX],
                    });

                    const opacity = burbuja.animation.interpolate({
                        inputRange: [ 0, 0.9, 1 ],
                        outputRange: [ 0, 0.6, 0 ],
                    });

                    return(
                        <Animated.Image
                            key={index}
                            source={ burbujaImage }
                            style={[
                                style.burbuja,
                                {
                                    opacity,
                                    transform: [
                                        { translateX },
                                        { translateY },
                                        { scale: burbuja.scale }
                                    ]
                                }
                            ]}
                        />
                    );

            }) }
            {/*Formulario*/}
            <View
                style={ style.loginBox }
            >
                <Text
                    style={ style.title }
                >
                    Login
                </Text>
                { (request == false) &&
                    <Text
                        style={{
                            ...style.title,
                            color: "red"
                        }}
                    >
                        Datos erroreos
                    </Text>
                }
                <TextInput
                    style={ style.input }
                    placeholder='username'
                    placeholderTextColor="#ccc"
                    value={ state.username }
                    onChangeText={ (text) => handleInputChange("username",text) }
                />
                <TextInput
                    style={ style.input }
                    placeholder='contraseña'
                    placeholderTextColor="#ccc"
                    secureTextEntry={true}
                    value={ state.password }
                    onChangeText={ (text) => handleInputChange("password",text) }
                />
                <View
                    style={{
                        justifyContent: "center",
                        alignItems: "center"
                    }}
                > 
                    { (!loading) &&
                        <BtnTouch
                            title="Ingresar"
                            background='lightblue'
                            bColor='black'
                            onPress={ () => handleLogin() }
                        />
                    }
                </View>
            </View>
        </ImageBackground>
    );
}

const style = StyleSheet.create({
  container: {
    flex: 1,
  },
  burbuja: {
    position: 'absolute',
    width: 100,
    height: 100,
  },
  loginBox: {
    position: 'absolute',
    bottom: 300,
    width: '80%',
    alignSelf: 'center',
    backgroundColor: 'rgba(255,255,255,0.8)',
    borderRadius: 20,
    padding: 20,
  },
  title: {
    color: 'black',
    fontSize: 28,
    marginBottom: 20,
    textAlign: 'center',
  },
  input: {
    backgroundColor: '#222',
    color: '#fff',
    borderRadius: 10,
    padding: 12,
    marginBottom: 15,
  }
});

