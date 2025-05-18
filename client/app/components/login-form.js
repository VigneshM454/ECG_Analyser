import Component from '@glimmer/component';
import { tracked } from '@glimmer/tracking';
import { action } from '@ember/object';
import axios from 'axios';
import { inject as service } from '@ember/service';
export default class LoginFormComponent extends Component {
  @service router;
  @service ecgData;

  @tracked isNewUser = true;

  @tracked demoUser = {
    name: '',
    email: '',
    phone: null,
    pass: '',
    confirmPass: '',
  };

  @action updateName(event) {
    this.demoUser.name = event.target.value;
    console.log(this.demoUser.name);
  }

  @action updateEmail(event) {
    this.demoUser.email = event.target.value;
  }

  @action updatePhone(event) {
    this.demoUser.phone = event.target.value;
  }

  @action updatePass(event) {
    this.demoUser.pass = event.target.value;
  }

  @action updateConfirmPass(event) {
    this.demoUser.confirmPass = event.target.value;
  }

  @action submitForm(event) {
    event.preventDefault();
    if (this.demoUser.pass != this.demoUser.confirmPass && this.isNewUser) {
      alert('Password and Confirm Password are not same');
    } else {
      console.log(this.demoUser);
      console.log(this.isNewUser);
      if (this.isNewUser) {
        axios
          .post(`${this.ecgData.baseUrl}/createAccount`, this.demoUser, {
            withCredentials: true,
          })
          .then((res) => {
            console.log(res);
            this.router.transitionTo('home');
            this.ecgData.updateAuth('true')
          })
          .catch((err) => {
            alert(err.response.data.msg);
            console.log(err);
          })
          .finally(() => (this.ecgData.isAuthChecked = true));
      } else {
        console.log('just a login');
        axios
          .post(
            `${this.ecgData.baseUrl}/login`,
            {
              email: this.demoUser.email,
              pass: this.demoUser.pass,
            },
            { withCredentials: true },
          )
          .then((res) => {
            console.log(res);
            console.log(res.data);
            this.router.transitionTo('home');
            this.ecgData.updateAuth('true')
          })
          .catch((err) => {
            alert(err.response.data.msg);
            console.log(err);
          })
          .finally(() => (this.ecgData.isAuthChecked = true));
      }
    }
  }
}
