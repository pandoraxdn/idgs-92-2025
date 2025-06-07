import { useEffect, useState } from "react";
import { pandoraApi } from "../api/pandoraApi";
import { UsersData } from "../interfaces/requestApi";
import { FormData } from "./useFormData";

interface UseUserApi{
    isLoading:  boolean;
    listUser:   FormData;
    loadUsers:  () => void;
    createUser: ( data: FormData ) => void;
    updateUser: ( data: FormData ) => void;
    deleteUser: ( data: FormData ) => void;
}

export const useUserApi = () => {

    const [ isLoading, setIsLoading ] = useState<boolean>(false);
    const [ listUser, setListUser ] = useState<UsersData>( {} as UsersData );

    const apiUrl: string = 'http://192.168.1.38:3000/api/v1/users';

    const loadUsers = async () => {
        setIsLoading( true );
        const response = await pandoraApi.get( apiUrl );
        setListUser(response.data);
        setIsLoading(false);
    }

    const createUser = async ( data: FormData ) => {
        const dataBody = {
            username: data.username,
            imagen: data.imagen,
            password: data.password,
            tipo: data.tipo,
        }

        await pandoraApi.post( apiUrl, dataBody );
    }

    const updateUser = async ( data: FormData ) => {

        if( data.password != '' ){
            const dataBody = {
                username: data.username,
                imagen: data.imagen,
                password: data.password,
                tipo: data.tipo,
            }
            await pandoraApi.patch( apiUrl + `/${data._id}`, dataBody );
    
        }else{
            const dataBody = {
                username: data.username,
                imagen: data.imagen,
                tipo: data.tipo,
            }

            await pandoraApi.patch( apiUrl + `/${data._id}`, dataBody );
        }
    
    }

    const deleteUser = async ( data: FormData ) => {
        await pandoraApi.delete( apiUrl + `/${data._id}`);
    }

    useEffect(() => {
        loadUsers();
    },[]);

    return { isLoading, listUser, loadUsers, createUser, updateUser, deleteUser };

}
