import React, { createContext, useReducer, ReactNode } from 'react';
import { authReducer } from './authReducer';

export interface AuthState{
    isLoggenIn: boolean;
    username:   string;
    avatar:     string;
}

export const AuthInitialState: AuthState = {
    isLoggenIn: false,
    username:   "",
    avatar:     "",
}

export interface AuthContextProps{
    authState:      AuthState;
    singIng:        () => void;
    logout:         () => void;
    changeAvatar:   ( avatar:string ) => void;
    changeUserName: ( username: string ) => void;
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

    return(
        <AuthContext.Provider
            value={{
                authState,
                singIng,
                logout,
                changeAvatar,
                changeUserName
            }}
        >
            { children }
        </AuthContext.Provider>
    );

}

