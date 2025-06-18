import React, { createContext, useReducer, ReactNode } from 'react';
import { authReducer } from './authReducer';

export interface AuthState{
    isLoggedIn: boolean;
    username:   string;
    avatar:     string;
    theme:      string;
}

export const AuthInitialState: AuthState = {
    isLoggedIn: false,
    username:   "",
    avatar:     "",
    theme:      "light",
}

export interface AuthContextProps{
    authState:      AuthState;
    singIng:        () => void;
    logout:         () => void;
    changeAvatar:   ( avatar:string ) => void;
    changeUserName: ( username: string ) => void;
    changeTheme:    ( theme: string ) => void;
}

export const AuthContext = createContext( {} as AuthContextProps );

export const AuthProvider = ( { children } : { children: ReactNode } ) => {

    const [ authState, dispatch ] = useReducer( authReducer, AuthInitialState );

    const singIng = () => dispatch({ type: "singIng" });

    const logout  = () => dispatch({ type: "logout" });

    const changeAvatar = ( avatar: string ) => {
        dispatch({ type: "changeAvatar", payload: avatar });
    }
    const changeUserName = ( username: string ) => {
        dispatch({ type: "changeUserName", payload: username });
    }

    const changeTheme = ( theme: string ) => {
        dispatch({ type: "changeTheme", payload: theme });
    }

    return(
        <AuthContext.Provider
            value={{
                authState,
                singIng,
                logout,
                changeAvatar,
                changeUserName,
                changeTheme
            }}
        >
            { children }
        </AuthContext.Provider>
    );

}

