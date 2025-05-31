import { Injectable } from '@nestjs/common';
import { User } from './schemas/user.schemas';
import { Model } from 'mongoose';
import { InjectModel } from '@nestjs/mongoose';
import { CreateUserDto } from './dto/create-user.dto';
import { UpdateUserDto } from './dto/update-user.dto';
import * as bcrypt from 'bcrypt';

@Injectable()
export class UsersService {
    constructor(
        @InjectModel( User.name ) private userModel: Model<User>
    ){}

    async login( user: UpdateUserDto ){
        try{
            const register: User = await this.userModel.findOne({ username: user.username }).exec();
            return ( await bcrypt.compare(user.password, register.password) ) ? register: false;
        }catch{
            return false;
        }
    }

    async create( user: CreateUserDto ) {
        const saltOrRounds = 10;
        const hash = await bcrypt.hash( user.password, saltOrRounds);
        const register = { ...user, password: hash };
        const created_user = new this.userModel( register );
        return created_user.save();
    }

    async findAll() {
        return this.userModel.find().exec();
    }

    async findOne(id: string) {
        return this.userModel.findById( id ).exec();
    }

    async update(id: string, user: UpdateUserDto) {

        ( user.password ) && ( async () => {    
            const saltOrRounds = 10;
            const hash = await bcrypt.hash( user.password, saltOrRounds);
            const register = { ...user, password: hash };
            return this.userModel.findByIdAndUpdate( id, register, { 
                new: true 
            }).exec();
        })();

        return this.userModel.findByIdAndUpdate( id, user, { 
            new: true 
        }).exec();
    }

    async remove(id: string) {
        return this.userModel.findByIdAndDelete(id).exec();
    }
}
