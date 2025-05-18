// app/routes/authenticated-route.js
import Route from '@ember/routing/route';
import { inject as service } from '@ember/service';

export default class AuthenticatedRoute extends Route {
  @service ecgData;
  @service router;

  async beforeModel() {
    const isAuth = this.ecgData.isAuthChecked
      ? this.ecgData.isAuthentic
      : await this.ecgData.checkAuth();

    if (!isAuth) {
      this.router.transitionTo('index');
    }
  }

  //   async beforeModel(transition) {
  //     // alert('im executed from authenticated')
  //   if (!this.ecgData.isAuthChecked) {
  //     const isAuth = await this.ecgData.checkAuth();

  //     if (!isAuth) {
  //       this.router.transitionTo('index');
  //     }
  //     // No need to manually transition to 'home'; you're already in the AuthenticatedRoute.
  //   } else if (!this.ecgData.isAuthentic) {
  //     this.router.transitionTo('index');
  //   }
  // }
}
