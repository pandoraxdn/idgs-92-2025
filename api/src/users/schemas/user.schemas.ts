import { Schema, Prop, SchemaFactory } from '@nestjs/mongoose';

export enum TypeUser{
    admin   = "admin",
    user    = "user",
    client  = "client"
}

@Schema()
export class User{
    @Prop({ require: true })
    username:   string;

    @Prop({ require: true })
    password:   string;

    @Prop({ require: true })
    imagen:     string;

    @Prop({ default: TypeUser.user })
    tipo:       TypeUser;
}

export const UserSchema = SchemaFactory.createForClass(User);
