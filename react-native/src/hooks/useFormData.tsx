import { useState } from "react";
import { useUserApi } from "./useUserApi";

export interface FormData{
    _id:        string;
    username:   string;
    password:   string;
    imagen:     string;
    tipo:       "user" | "admin" | "client";
}

interface UseFormHook{
    formData: FormData;
    formList: FormData[];
    handleInputChange: ( fieldName: keyof FormData, value: string | number ) => void;
    handleSubmit: () => void;
    handleDelete: () => void;
}

export const useFormHook = (): UseFormHook => {

    const { createUser, updateUser, deleteUser } = useUserApi();

    const initialState: FormData = {
        _id:        '',
        username:   '',
        password:   '',
        imagen:     '',
        tipo:       'user',
    }

    const [ formData, setFormData ] = useState<FormData>(initialState);
    const [ formList, setFormList ] = useState<FormData[]>([]);
    
    const handleInputChange = ( fieldName: keyof FormData, value: string | number ) => {
        setFormData( (prevData) => ({
            ...prevData,
            [fieldName]: value
        }));
    }

    const handleSubmit = () => {
        setFormList( (prevList) => [...prevList, formData] );

        ( formData._id !== '' )
        ? updateUser(formData)
        : createUser( formData );

        setFormData( initialState );
    }

    const handleDelete = () => {
        deleteUser( formData );
        setFormData( initialState );
    }

    return {
        formData,
        formList,
        handleInputChange,
        handleSubmit,
        handleDelete
    }
}

