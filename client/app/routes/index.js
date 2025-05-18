import Route from '@ember/routing/route';
import { inject as service } from '@ember/service';

export default class IndexRoute extends Route {
  @service ecgData;
  @service router;

  async beforeModel() {
    console.log('isAuthChecked', this.ecgData.isAuthChecked);
    const isAuth = this.ecgData.isAuthChecked
      ? this.ecgData.isAuthentic
      : await this.ecgData.checkAuth();
    console.log('isAuths ' + isAuth);
    if (isAuth) {
      this.router.transitionTo('home');
      console.log('in index trying to redirect user to home');
    }
  }

  // async beforeModel(transition) {
  //   // alert('im executed from index')
  //   if (!this.ecgData.isAuthChecked) {
  //     const isAuth = await this.ecgData.checkAuth();

  //     if (isAuth) {
  //       this.router.transitionTo('home');
  //     }
  //     // No need to manually transition to 'home'; you're already in the AuthenticatedRoute.
  //   } else if (this.ecgData.isAuthentic) {
  //     this.router.transitionTo('home');
  //   }

  //   // if(this.ecgData.isAuthentic)
  //   // if(this.ecgData.isAuthChecked ){
  //   //   if(this.ecgData.isAuthentic){
  //   //     this.router.transitionTo('home')
  //   //   }
  //   //   else{
  //   //     const isAuth = await this.ecgData.checkAuth();

  //   //   }
  //   //   // this.router.transitionTo('home')
  //   // }

  // }
}

// if (!this.ecgData.isAuthChecked) {
//   await this.ecgData.checkAuth();
// }
// if (!this.ecgData.isAuthentic) {
//   this.router.transitionTo('index');
// }
