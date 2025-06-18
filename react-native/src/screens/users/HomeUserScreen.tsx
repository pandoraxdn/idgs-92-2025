import React, { useEffect, useContext } from 'react';
import { View, Text, FlatList, RefreshControl } from 'react-native';
import { useNavigation, useIsFocused } from '@react-navigation/native';
import { useUserApi } from '../../hooks/useUserApi';
import { BtnTouch } from '../../components/BtnTouch';
import { CardUser } from '../../components/CardUser';
import { appTheme } from '../../themes/appTheme';
import { UsersData } from '../../interfaces/requestApi';
import { AuthContext } from '../../context/AuthContext';

export const HomeUserScreen = () => {

    const { singIng, logout, authState } = useContext( AuthContext );

    const isFocused = useIsFocused();

    useEffect(() => {
        loadUsers();
    },[isFocused]);

    const { listUser, isLoading, loadUsers } = useUserApi();
    const navigation = useNavigation();

    const create_user: UsersData = {
        _id:        '',
        username:   '',
        password:   '',
        tipo:       'user',
        imagen:     '',
        __v:        0,
    }

    return(
        <View
            style={{
                ...appTheme.container,
                marginTop: 20
            }}
        >
            <FlatList
                data={ Object.values(listUser) }
                keyExtractor={ (item) => '#'+item._id }
                ListHeaderComponent={(
                    <View>
                        {
                            ( authState.isLoggenIn )
                            ? <BtnTouch 
                                title='Cerrar Sesión'
                                onPress={ () => logout() }
                                bColor="black"
                            />
                            : <BtnTouch 
                                title='Iniciar Sesión'
                                onPress={ () => singIng() }
                                bColor="blue"
                            />
                        }
                        <Text
                            style={{
                                ...appTheme.text,
                                fontSize: 30
                            }}
                        >
                            Lista de Usuarios
                        </Text>
                        <BtnTouch
                            title='Crear usuario'
                            onPress={ () => navigation.navigate("FormScreen",{ user: create_user }) }
                            background='violet'
                            bColor='pink'
                        />
                    </View>
                )}
                refreshControl={
                    <RefreshControl
                        refreshing={ (isLoading) }
                        onRefresh={ loadUsers }
                        colors={[ "pink", "violet" ]}
                        progressBackgroundColor="black"
                    />
                }
                showsVerticalScrollIndicator={false}
                numColumns={2}
                renderItem={ ({ item }) => (
                    <CardUser
                        user={item}
                    />    
                )}
            />
        </View>
    );
}
